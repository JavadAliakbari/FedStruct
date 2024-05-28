# To run the code
1. Clone the repository to your *local_directory* with:
    > git clone https://github.com/JavadAliakbari/FedStruct.git *local_directory*

2. Run the following lines to download required packages:  
    >conda env create -f ./FedStruct.yml  
    >conda activate FedStruct  

3. You can change hyper-parameters in **~/config/config_*{dataset_name}*.py** according to different testing scenarios.  
*{dataset_name}* can be **Cora**, **CiteSeer**, **PubMed**, **chameleon**, **Photo**, **Amazon-ratings**;

4. Run the main file with  
    > **CONFIG_PATH="./config/config_{dataset_name}.yml" python src/main.py**  

    You can access results in **results/{dataset_name}/**
5. You can also run the simulation file with  
    > **CONFIG_PATH="./config/config_{dataset_name}.yml" python src/simulations/simulation.py**

    You can access results in **results/Simulation/{dataset_name}/**
