import pandas as pd

file_path = r"C:\Users\andre\PycharmProjects\Lotto\Repo.xlsx"

def read_excel_and_print_row_count(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Check if the DataFrame contains any data
        if not df.empty:
            row_count = df.shape[0]
            print(f"The Excel file contains {row_count} rows filled with numbers.")
        else:
            print("The Excel file is empty.")
    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    read_excel_and_print_row_count(file_path)
