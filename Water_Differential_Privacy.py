import pandas as pd
import numpy as np

def add_noise(value, epsilon):
    scale = 1 / epsilon
    noise = np.random.laplace(scale=scale)
    return value + noise

def apply_differential_privacy(input_file, output_file, epsilon):
    # TODO
    df = pd.read_csv(input_file, delimiter = ';')
    
    df['meter.reading'] = df['meter.reading'].apply(lambda x: add_noise(x, epsilon))
        
    df.to_csv(output_file, index = False)


def main():
    input_file = 'water_data.csv'
    output_file = 'output.csv'
    epsilon = 1.0  # Privacy budget
    apply_differential_privacy(input_file, output_file, epsilon)
    print("Differential privacy applied successfully to the CSV file.")

if __name__ == "__main__":
    main()
