import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn.models import Node2Vec


def train(model: Node2Vec, loader, optimizer: torch.optim.SparseAdam):
    model.train()
    total_loss = 0
    count = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / len(loader)


@torch.no_grad()
def test(model, graph):
    model.eval()
    z = model()
    acc = model.test(
        z[graph.train_mask],
        graph.y[graph.train_mask],
        z[graph.test_mask],
        graph.y[graph.test_mask],
        max_iter=100,
    )
    return acc


def find_node2vect_embedings(edge_index, epochs=50, embedding_dim=64, plot=False):
    model = Node2Vec(
        edge_index,
        embedding_dim=embedding_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
    )

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    bar = tqdm(total=epochs)
    res = []
    for epoch in range(epochs):
        loss = train(model, loader, optimizer)
        bar.set_description(f"Epoch: {epoch:02d}")
        bar.set_postfix({"Loss": loss})
        bar.update()

        res.append(loss)

    # print(f"test accuracy: {test(model, )}")
    if plot:
        plt.plot(res)

    model.eval()
    z = model()
    z = z.detach()

    return z
