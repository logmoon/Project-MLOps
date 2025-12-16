"""
Fixed Elasticsearch connection for ES 8.x
Replace the get_elasticsearch_client function in model_pipeline.py
"""

from elasticsearch import Elasticsearch
from datetime import datetime
import warnings

def get_elasticsearch_client():
    """
    Create and return Elasticsearch client connection
    Compatible with Elasticsearch 8.x
    
    Returns:
        Elasticsearch: Connected ES client or None if connection fails
    """
    try:
        # For ES 8.x with security disabled
        es = Elasticsearch(
            ["http://localhost:9200"],
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Test connection
        if es.ping():
            print("✅ Connected to Elasticsearch!")
            
            # Get cluster info
            info = es.info()
            print(f"   Cluster: {info['cluster_name']}")
            print(f"   Version: {info['version']['number']}")
            
            # Create index if it doesn't exist
            index_name = "mlflow-metrics"
            if not es.indices.exists(index=index_name):
                es.indices.create(
                    index=index_name,
                    mappings={
                        "properties": {
                            "run_id": {"type": "keyword"},
                            "timestamp": {"type": "date"},
                            "model_name": {"type": "keyword"},
                            "metrics": {"type": "object"},
                            "params": {"type": "object"},
                            "tags": {"type": "object"}
                        }
                    }
                )
                print(f"   Created index: {index_name}")
            else:
                print(f"   Using existing index: {index_name}")
            
            return es
        else:
            print("❌ Failed to ping Elasticsearch")
            return None
            
    except Exception as e:
        print(f"⚠️ Elasticsearch connection error: {e}")
        print("   Elasticsearch may not be running. Start it with: make monitoring-up")
        return None


def log_to_elasticsearch(run_id, metrics, params, tags=None, model_name="water_potability"):
    """
    Send MLflow run data to Elasticsearch (ES 8.x compatible)
    
    Args:
        run_id (str): MLflow run ID
        metrics (dict): Model metrics (accuracy, precision, etc.)
        params (dict): Model parameters and hyperparameters
        tags (dict): Additional tags for the run
        model_name (str): Name of the model
    
    Returns:
        bool: True if successfully logged, False otherwise
    """
    if es_client is None:
        print("⚠️ Elasticsearch not connected, skipping logging")
        return False
    
    try:
        document = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": metrics,
            "params": params,
            "tags": tags or {}
        }
        
        # ES 8.x uses 'document' parameter instead of 'body'
        response = es_client.index(
            index="mlflow-metrics",
            document=document
        )
        
        print(f"✅ Logged to Elasticsearch: {response['result']} (ID: {response['_id']})")
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to log to Elasticsearch: {e}")
        return False


# Test function to verify connection
def test_es_connection():
    """
    Test Elasticsearch connection and functionality
    """
    print("\n" + "="*60)
    print("TESTING ELASTICSEARCH CONNECTION")
    print("="*60 + "\n")
    
    es = get_elasticsearch_client()
    
    if es:
        # Try to index a test document
        try:
            test_doc = {
                "run_id": "test_run_123",
                "timestamp": datetime.now().isoformat(),
                "model_name": "test_model",
                "metrics": {"accuracy": 0.95},
                "params": {"test_param": "value"},
                "tags": {"test": "true"}
            }
            
            response = es.index(
                index="mlflow-metrics",
                document=test_doc
            )
            print(f"✅ Successfully indexed test document!")
            print(f"   Result: {response['result']}")
            print(f"   Document ID: {response['_id']}")
            
            # Search for the document
            search_result = es.search(
                index="mlflow-metrics",
                query={"match": {"run_id": "test_run_123"}}
            )
            
            print(f"✅ Successfully searched documents!")
            print(f"   Found {search_result['hits']['total']['value']} documents")
            
            print("\n✅ All tests passed! Elasticsearch is working correctly.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("❌ Could not connect to Elasticsearch")
        print("\nTroubleshooting:")
        print("1. Verify ES is running: curl http://localhost:9200")
        print("2. Check Docker: docker ps | grep elasticsearch")
        print("3. Check logs: docker logs elasticsearch")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # For testing this file directly
    es_client = get_elasticsearch_client()
    test_es_connection()
