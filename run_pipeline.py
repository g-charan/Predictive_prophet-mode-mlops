from pipelines.train_pipeline import train_pipeline2
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path = r"D:\Intern\MLOP\Test_project\data\expense_data_1.csv"
    rmse = train_pipeline2(data_path)
    
#mlflow ui --backend-store-uri "file:C:\Users\cherr\AppData\Roaming\zenml\local_stores\76c8c936-fce8-4284-96d9-61f769a407dd\mlruns"