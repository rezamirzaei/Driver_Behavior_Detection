"""Test the classification module."""
import sys
from pathlib import Path

# Output file
output_file = Path(__file__).parent.parent / 'results' / 'test_output.txt'

def log(msg):
    print(msg)
    with open(output_file, 'a') as f:
        f.write(msg + '\n')

# Clear output file
with open(output_file, 'w') as f:
    f.write('')

log('Testing src/classification module...')

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.classification import (
        load_or_build_dataset,
        prepare_classification_data,
        get_all_classifiers,
        train_single_classifier,
        get_feature_importance,
        MCPLogisticRegression,
        SCADLogisticRegression,
    )
    log('✅ Imports successful')
except Exception as e:
    log(f'❌ Import error: {e}')
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

try:
    # Load data
    df = load_or_build_dataset(
        data_dir=Path('data/UAH-DRIVESET-v1'),
        cache_path=Path('data/processed/uah_raw_features.csv')
    )
    log(f'✅ Loaded: {df.shape}')

    # Prepare data
    data = prepare_classification_data(df, test_drivers=['D6'])
    log(f'✅ Train: {data.X_train.shape}, Test: {data.X_test.shape}')

    # Get classifiers
    classifiers = get_all_classifiers()
    log(f'✅ Classifiers: {len(classifiers)}')

    # Train test models
    from sklearn.linear_model import LogisticRegression
    test_models = {
        'Logistic (L2)': LogisticRegression(max_iter=1000),
        'Logistic (MCP)': MCPLogisticRegression(alpha=0.1, gamma=3.0),
    }

    log('\nTraining test models...')
    for name, model in test_models.items():
        result = train_single_classifier(model, name, data)
        log(f'  {name}: Train={result.train_accuracy:.3f}, Test={result.test_accuracy:.3f}')

    log('\n✅ All tests passed!')

except Exception as e:
    log(f'❌ Error: {e}')
    import traceback
    log(traceback.format_exc())
