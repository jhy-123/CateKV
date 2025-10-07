import os
import random
import numpy as np
import pandas as pd

def generate_final_matrices(
    main_dir='path/to/main_dir',
    output_dir='/path/to/output',
    alphas=[0.3],
    files_per_subfolder=50,
    seed=None
):
    """
    Randomly select a specified number of CSV files from each subfolder,
    calculate the frequency of 1s at each position, and generate final
    binary (0-1) matrices for multiple alpha parameters. Each resulting
    matrix is saved as a separate file.

    Parameters:
    - main_dir (str): The main directory containing multiple subfolders.
      Each subfolder should contain multiple CSV files.
    - output_dir (str): The main directory where the final matrices will be saved.
    - alphas (list of float): A list of values between 0 and 1 used to determine
      the top proportion of positions to set as 1.
    - files_per_subfolder (int): The number of CSV files to randomly select from
      each subfolder.
    - seed (int, optional): Random seed for reproducibility.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get all subfolders under the main directory
    subfolders = [os.path.join(main_dir, d) for d in os.listdir(main_dir)
                 if os.path.isdir(os.path.join(main_dir, d))]

    if len(subfolders) == 0:
        raise ValueError(f"No subfolders found under main directory {main_dir}.")

    for subfolder in subfolders:
        selected_matrices = []
        sub_name = subfolder.split("/")[-1]
        # Get all CSV files in the subfolder
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
        
        if len(csv_files) < files_per_subfolder:
            raise ValueError(f"Subfolder {subfolder} contains fewer than {files_per_subfolder} CSV files.")

        save_dir = os.path.join(output_dir, sub_name)
        # Ensure that the output directory exists
        os.makedirs(save_dir, exist_ok=True)
    
        # Randomly select the specified number of CSV files
        selected_files = random.sample(csv_files, files_per_subfolder)
        
        for alpha in alphas:
            if not (0 < alpha < 1):
                raise ValueError(f"Alpha value {alpha} must be between (0, 1).")
            for file in selected_files:
                file_path = os.path.join(subfolder, file)
                # Read the CSV file and convert it to float values
                matrix = pd.read_csv(file_path, header=None).values.astype(float)
                threshold = np.quantile(matrix, alpha)
                matrix = (matrix <= threshold).astype(int)
                selected_matrices.append(matrix)

            # Sum all selected matrices to get the frequency of 1s at each position
            sum_matrix = np.sum(selected_matrices, axis=0)
   
            # Compute the threshold: the top alpha proportion of highest values
            flat_sum = sum_matrix.flatten()
            num_elements = flat_sum.size
            num_top = int(np.ceil(alpha * num_elements))
            
            # Find the smallest value among the top num_top largest elements as the threshold
            threshold = np.partition(flat_sum, -num_top)[-num_top]
            
            # Generate the final 0-1 matrix
            final_matrix = (sum_matrix >= threshold).astype(int)
            
            # Define the output file path, including the alpha value
            output_file_path = os.path.join(save_dir, f"head_type_p{int(alpha*100)}.csv")
            
            # Save the final matrix as a CSV file
            pd.DataFrame(final_matrix).to_csv(output_file_path, header=False, index=False)
            
            print(f"The 0-1 matrix for alpha={alpha} has been saved to: {output_file_path}")

if __name__ == "__main__":
    # Example usage with multiple alpha values
    alphas = [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  # Adjust as needed
    generate_final_matrices(
        main_dir='/select_headmask/headmask_newckpt/Llama-3-8B-Instruct-Gradient-1048k/vt_fraction0.998_top1.0_lastq64',
        output_dir='/select_headmask/headmask_newckpt/Llama-3-8B-Instruct-Gradient-1048k/vt_fraction0.998_top1.0_lastq64/final_metric',
        alphas=alphas,
        files_per_subfolder=50,
        seed=42  # Optional, set a random seed for reproducibility
    )
