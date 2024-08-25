from zenml import step

@step
def deployment_trigger(accuracy: float,min_accuracy:float = 0.9) -> bool:
    """Triggers deployment if accuracy is above a certain threshold."""
    if accuracy >= min_accuracy:
        return True
    else:
        return False