
# read 5 csv files and plot line graphs for each file on the same plot
import pandas as pd
import matplotlib.pyplot as plt 

# print(plt.rcParams['axes.prop_cycle'])

def plot_prompt_sensitivity(file_paths, labels, output_path):   
    plt.figure(figsize=(10, 6))

    # color card
    colors = ["#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#8c564b',]
    markers = ['o', 's', '^', 'D', 'v']

    for file_path, label in zip(file_paths, labels):
        # Read the CSV file
        df = pd.read_csv(file_path)

        # add ber 1e-10 before 1e-9, the acc is the same as 1e-9
        df = pd.concat([pd.DataFrame({'ber': [1e-10], 'acc': [df['acc'][0]]}), df], ignore_index=True)
        
        # Assuming the CSV has columns 'prompt_length' and 'accuracy'
        plt.plot(df['ber'], df['acc'], linestyle='--', linewidth=2, marker=markers.pop(0), label=label,color=colors.pop(0))
    
    # set x labels to 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3
    
    
    # set x to log scale
    plt.xscale('log')   

    # set y limit from 0 to 100
    plt.ylim(0, 100)
    # set y axis steps of 10
    plt.yticks(range(0, 101, 20))

    plt.xticks([1e-10, 1e-9, 1e-8, 5e-8, 1e-7, 3e-7, 5e-7, 1e-6],
               ["1e-10", "1e-9", "1e-8", " ", "1e-7", " ", " ", "1e-6"])


    # set font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18) 

    # plt.title('Prompt Sensitivity Analysis')
    plt.xlabel('Bit Error Rate', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    file_paths = [
        '/home/bobzhou/FAPT/results/oxford_flowers/ZeroshotCLIP/vit_l14/zeroshot/resilience_component_wise_image_none.csv',
        '/home/bobzhou/FAPT/results/oxford_flowers/ZeroshotCLIP/vit_l14/zeroshot/resilience_component_wise_image_hep.csv',
        '/home/bobzhou/FAPT/results/oxford_flowers/ZeroshotCLIP/vit_l14/zeroshot/resilience_component_wise_image_hep2.csv',
        '/home/bobzhou/FAPT/results/oxford_flowers/ZeroshotCLIP/vit_l14/zeroshot/resilience_component_wise_image_alpha.csv',
        '/home/bobzhou/FAPT/results/oxford_flowers/ZeroshotCLIP/vit_l14/zeroshot/resilience_component_wise_image_num.csv'
    ]

    labels = ['P1: [class]', 
              'P2: A photo of a [class]', 
              'P3: An image of a [class]', 
              'P4: dfaf akfj lajd [class]', 
              'P5: 3232 3434 5698 [class]'
    ]
    
    output_path = 'results/prompt_sensitivity_analysis.png'
    
    plot_prompt_sensitivity(file_paths, labels, output_path)