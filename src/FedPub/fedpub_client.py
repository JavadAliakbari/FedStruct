from datetime import datetime
import logging
import time

import torch
import torch.nn.functional as F

from src.utils.utils import calc_accuracy
from src.FedPub.utils import *
from src.FedPub.nets import *
from src.utils.config_parser import Config
from src.utils.graph import Graph

path = os.environ.get("CONFIG_PATH")
config = Config(path)


class FedPubClient:
    def __init__(
        self,
        graph: Graph,
        id,
        proxy,
        save_path="./",
        logger=None,
    ):
        self.graph = graph
        self.id = id
        self.save_path = save_path
        self.LOGGER = logger or logging
        # self.loader = PubDataLoader(self.graph)
        self.model = MaskedGCN(
            graph.num_features,
            config.fedpub.n_dims,
            graph.num_classes,
            config.fedpub.l1,
        )
        self.parameters = list(self.model.parameters())
        self.proxy = proxy
        self.init_state()

    def init_state(self):
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=config.fedpub.lr,
            weight_decay=config.fedpub.weight_decay,
        )
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"{self.save_path}/logs/{now}"
        os.makedirs(self.log_path, exist_ok=True)
        self.reset_log()

    def reset_log(self):
        self.log = {
            "lr": [],
            "train_lss": [],
            "ep_local_val_lss": [],
            "ep_local_val_acc": [],
            "rnd_local_val_lss": [],
            "rnd_local_val_acc": [],
            "ep_local_test_lss": [],
            "ep_local_test_acc": [],
            "rnd_local_test_lss": [],
            "rnd_local_test_acc": [],
            "rnd_sparsity": [],
            "ep_sparsity": [],
        }

    def update(self, state_dict):
        self.prev_w = convert_np_to_tensor(state_dict)
        set_state_dict(self.model, state_dict, skip_stat=True, skip_mask=True)

    def get_train_results(self, curr_rnd):
        self.curr_rnd = curr_rnd
        results = self.train()
        data = self.transfer_to_server()

        return data, results

    def report_result(self, result, framework=""):
        self.LOGGER.info(f"{framework} results for client{self.id}:")
        self.LOGGER.info(f"{result}")

    def get_sparsity(self):
        n_active, n_total = 0, 1
        for mask in self.masks:
            pruned = torch.abs(mask) < config.fedpub.l1
            mask = torch.ones(mask.shape).masked_fill(pruned, 0)
            n_active += torch.sum(mask)
            _n_total = 1
            for s in mask.shape:
                _n_total *= s
            n_total += _n_total
        return ((n_total - n_active) / n_total).item()

    def train(self):
        self.masks = []
        for name, param in self.model.state_dict().items():
            if "mask" in name:
                self.masks.append(param)

        for ep in range(config.model.local_epochs):
            # st = time.time()
            self.model.train()
            # for _, batch in enumerate(self.loader.pa_loader):
            self.optimizer.zero_grad()
            y_hat = self.model(self.graph)

            train_acc = calc_accuracy(
                y_hat[self.graph.train_mask].argmax(dim=1),
                self.graph.y[self.graph.train_mask],
            )
            train_loss = F.cross_entropy(
                y_hat[self.graph.train_mask], self.graph.y[self.graph.train_mask]
            )

            #################################################################
            for name, param in self.model.state_dict().items():
                if "mask" in name:
                    train_loss += torch.norm(param.float(), 1) * config.fedpub.l1
                elif "conv" in name or "clsif" in name:
                    if self.curr_rnd == 0:
                        continue
                    train_loss += (
                        torch.norm(param.float() - self.prev_w[name], 2)
                        * config.fedpub.loc_l2
                    )
            #################################################################

            train_loss.backward()
            self.optimizer.step()

            sparsity = self.get_sparsity()
            val_acc, val_loss = self.validate(mode="valid")
            # test_acc, test_loss = self.validate(mode="test")
            # self.LOGGER.info(
            #     f"[c: {self.client_id}], rnd:{self.curr_rnd+1}, ep:{ep}, "
            #     + f"val_local_loss: {val_local_lss.item():.4f}, val_local_acc: {val_local_acc:.4f}, lr: {self.get_lr()} ({time.time()-st:.2f}s)"
            # )
            self.log["train_lss"].append(train_loss.item())
            self.log["ep_local_val_acc"].append(val_acc)
            self.log["ep_local_val_lss"].append(val_loss)
            # self.log["ep_local_test_acc"].append(test_acc)
            # self.log["ep_local_test_lss"].append(test_loss)
            self.log["ep_sparsity"].append(sparsity)
        self.log["rnd_local_val_acc"].append(val_acc)
        self.log["rnd_local_val_lss"].append(val_loss)
        # self.log["rnd_local_test_acc"].append(test_acc)
        # self.log["rnd_local_test_lss"].append(test_loss)
        self.log["rnd_sparsity"].append(sparsity)
        self.save_log()

        result = {
            "Train Loss": round(train_loss.item(), 4),
            "Train Acc": round(train_acc, 4),
            "Val Loss": round(val_loss.item(), 4),
            "Val Acc": round(val_acc, 4),
        }

        return result

    @torch.no_grad()
    def get_functional_embedding(self):
        self.model.eval()
        with torch.no_grad():
            proxy_in = self.proxy
            proxy_out = self.model(proxy_in, is_proxy=True)
            proxy_out = proxy_out.mean(dim=0)
            proxy_out = proxy_out.clone().detach().cpu().numpy()
        return proxy_out

    def transfer_to_server(self):
        res = {
            "model": get_state_dict(self.model),
            "train_size": sum(self.graph.train_mask),
            "functional_embedding": self.get_functional_embedding(),
        }

        return res

    def set_weights(self, state_dict, skip_stat=False, skip_mask=False):
        set_state_dict(self.model, state_dict, skip_stat, skip_mask)

    def get_weights(self):
        return {"model": get_state_dict(self.model)}

    @torch.no_grad()
    def validate(self, mode="test"):
        # loader = self.loader.pa_loader

        with torch.no_grad():
            target, pred, loss = [], [], []
            # for batch in loader:
            mask = self.graph.test_mask if mode == "test" else self.graph.val_mask
            y_hat, lss = self.validation_step(self.graph, mask)
            pred.append(y_hat[mask])
            target.append(self.graph.y[mask])
            loss.append(lss)
            acc = calc_accuracy(
                torch.stack(pred).view(-1, self.graph.num_classes).argmax(dim=1),
                torch.stack(target).view(-1),
            )
        return acc, np.mean(loss)

    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.model.eval()
        y_hat = self.model(batch)
        if torch.sum(mask).item() == 0:
            return y_hat, 0.0
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    def get_test_results(self):
        test_acc, ـ = self.test_classifier()
        result = {"Test Acc": round(test_acc, 4)}

        return result

    def test_classifier(self):
        test_acc, test_loss = self.validate(mode="test")
        # return self.classifier.calc_test_accuracy(metric)
        return test_acc, test_loss

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_log(self):
        save(
            self.log_path,
            f"client_{self.id}.json",
            {"log": self.log},
        )
