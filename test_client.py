from mcp.client import Client

def test_server():
    # Create MCP client
    client = Client()

    # Test load_data tool
    print("Testing load_data tool...")
    try:
        result = client.load_data(file_path="test.csv")
        print("Result:", result)
    except Exception as e:
        print("Error:", str(e))

    # Test train_model tool
    print("\nTesting train_model tool...")
    try:
        result = client.train_model(
            file_path="test.csv",
            target_column="target",
            features=["feature1", "feature2"],
            model_type="random_forest"
        )
        print("Result:", result)
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_server()
