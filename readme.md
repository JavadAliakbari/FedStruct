# To run the code
1. Unzip the package to your local directory;
2. Run the following lines to download required packages:  
  conda env create -f ./FedStruct.yml
  conda activate FedStruct  
3. You can change hyper-parameters in ~/config/config_{dataset_name}.py according to different testing scenarios. {dataset_name} can be Cora, CiteSeer, PubMed, chameleon, Photo, Amazon-ratings;
4. Run the whole pipline with 'CONFIG_PATH="./config/config_{dataset_name}.yml" python main.py'.
5. You can access results in results/{dataset_name}/
