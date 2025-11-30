from mlflow.tracking import MlflowClient

def rollback_model(model_name, target_version):
    '''Rollback to specific model version'''
    client = MlflowClient()
    
    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage='Production',
        archive_existing_versions=True
    )
    
    print(f'✅ Rolled back {model_name} to version {target_version}')
    return target_version

if __name__ == '__main__':
    # Example usage
    rollback_model('ledgerx_quality_model', 11)
