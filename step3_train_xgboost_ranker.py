import xgboost as xgb
import numpy as np
import pickle
import os
import json
from sklearn.metrics import ndcg_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


class XGBoostRanker:
    def __init__(self, output_dir='checkpoints'):
        self.model = None
        self.feature_names = None
        self.best_iteration = 0
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("XGBOOST RANKER INITIALIZED")
        print("="*70)
        print(f"Output directory: {output_dir}")
    
    def load_data(self, train_path, val_path, test_path=None):
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        print("\n1. Loading training data...")
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        train_features = train_data['features']
        train_labels = train_data['labels']
        train_groups = train_data['groups']
        self.feature_names = train_data['feature_names']
        
        print(f"   Train features: {train_features.shape}")
        print(f"   Train labels: {train_labels.shape}")
        print(f"   Train queries: {len(train_groups)}")
        
        print("\n2. Loading validation data...")
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        val_features = val_data['features']
        val_labels = val_data['labels']
        val_groups = val_data['groups']
        
        print(f"   Val features: {val_features.shape}")
        print(f"   Val labels: {val_labels.shape}")
        print(f"   Val queries: {len(val_groups)}")
        
        print("\n3. Creating DMatrix...")
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dtrain.set_group(train_groups)
        
        dval = xgb.DMatrix(val_features, label=val_labels)
        dval.set_group(val_groups)
        
        print("   DMatrix created for train and val")
        
        dtest = None
        if test_path and os.path.exists(test_path):
            print("\n4. Loading test data...")
            with open(test_path, 'rb') as f:
                test_data = pickle.load(f)
            
            test_features = test_data['features']
            test_labels = test_data['labels']
            test_groups = test_data['groups']
            
            print(f"   Test features: {test_features.shape}")
            print(f"   Test labels: {test_labels.shape}")
            print(f"   Test queries: {len(test_groups)}")
            
            dtest = xgb.DMatrix(test_features, label=test_labels)
            dtest.set_group(test_groups)
            
            print("   DMatrix created for test")
        
        return dtrain, dval, dtest
    
    def train(self, 
              train_path='xgboost_features/train_features.pkl',
              val_path='xgboost_features/val_features.pkl',
              test_path='xgboost_features/test_features.pkl',
              num_boost_round=500,
              early_stopping_rounds=50):
        
        
        dtrain, dval, dtest = self.load_data(train_path, val_path, test_path)
        

        print("\n" + "="*70)
        print("XGBOOST PARAMETERS")
        print("="*70)
        
        params = {
            # Ranking objective
            'objective': 'rank:ndcg',       
            'eval_metric': ['ndcg@10', 'ndcg@20', 'map@10'],
            
            # Learning parameters
            'eta': 0.1,                     
            'max_depth': 6,                 
            'min_child_weight': 1,          
            'subsample': 0.8,               
            'colsample_bytree': 0.8,        
            'gamma': 0.1,                   
            
            # Regularization
            'lambda': 1.0,                  
            'alpha': 0.0,                   
            
            # System
            'tree_method': 'hist',          
            'device': 'cuda',               
            'nthread': -1,                  
            
            # Misc
            'seed': 42
        }
        
        
        for key, value in params.items():
            print(f"  {key:20s}: {value}")
        
        
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        print(f"Max iterations: {num_boost_round}")
        print(f"Early stopping: {early_stopping_rounds} rounds")
        print()
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        if dtest is not None:
            evals.append((dtest, 'test'))
        
        evals_result = {}
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        
        self.best_iteration = self.model.best_iteration
        
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best iteration: {self.best_iteration}")
        
        
        for split in ['train', 'val']:
            if split in evals_result:
                print(f"\n{split.upper()} Results (best iteration):")
                for metric in evals_result[split].keys():
                    best_score = evals_result[split][metric][self.best_iteration]
                    print(f"  {metric:15s}: {best_score:.4f}")
        
        
        self._save_model()
        
        
        self._analyze_feature_importance()
        
       
        self._plot_training_curves(evals_result)
        
        return self.model
    
    def _save_model(self):
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        model_path = os.path.join(self.output_dir, 'xgboost_ranker.json')
        self.model.save_model(model_path)
        print(f"Model saved: {model_path}")
        
        feature_path = os.path.join(self.output_dir, 'xgboost_feature_names.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Feature names saved: {feature_path}")
        

        metadata = {
            'best_iteration': self.best_iteration,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
    
    def _analyze_feature_importance(self):
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (Top 20)")
        print("="*70)
        
        importance = self.model.get_score(importance_type='weight')
        
        feature_importance = []
        for i, name in enumerate(self.feature_names):
            if f'f{i}' in importance:
                feature_importance.append((name, importance[f'f{i}']))
        
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        

        for i, (name, score) in enumerate(feature_importance[:20], 1):
            print(f"  {i:2d}. {name:40s}: {score:8.0f}")
        

        importance_path = os.path.join(self.output_dir, 'feature_importance.pkl')
        with open(importance_path, 'wb') as f:
            pickle.dump(feature_importance, f)
        
        print(f"\nFull feature importance saved: {importance_path}")
        

        importance_json_path = os.path.join(self.output_dir, 'feature_importance.json')
        importance_dict = {name: float(score) for name, score in feature_importance}
        with open(importance_json_path, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        
        print(f"Feature importance (JSON) saved: {importance_json_path}")
    
    def _plot_training_curves(self, evals_result):
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 14,              
            'axes.titlesize': 18,          
            'axes.labelsize': 16,          
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })

        print("\n" + "="*70)
        print("PLOTTING TRAINING CURVES")
        print("="*70)
        
        metrics = list(evals_result['train'].keys())
        
        for metric in metrics:
            plt.figure(figsize=(6, 5))
            
            for split in ['train', 'val', 'test']:
                if split in evals_result and metric in evals_result[split]:
                    values = evals_result[split][metric]
                    epochs = range(len(values))
                    plt.plot(epochs, values, label=split.capitalize(), linewidth=2)
            
            if 'val' in evals_result and metric in evals_result['val']:
                val_values = evals_result['val'][metric]
                best_idx = np.argmax(val_values)
                best_value = val_values[best_idx]
                
                plt.axvline(x=best_idx, color='red', linestyle='--', 
                            alpha=0.5, label='Best', linewidth=1)
                plt.scatter([best_idx], [best_value], color='red', 
                            s=100, zorder=5, marker='*')
            
            plt.xlabel('Iteration', fontsize=14, fontname='Times New Roman')
            plt.ylabel(metric.upper(), fontsize=14, fontname='Times New Roman')
            plt.title(f'{metric.upper()}', fontsize=16, fontweight='bold', fontname='Times New Roman')
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            

            plot_path = os.path.join(self.output_dir, f'training_curve_{metric}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Training curve saved: {plot_path}")

    
    def evaluate(self, test_path='xgboost_features/test_features.pkl'):
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)
        
        
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        
        test_features = test_data['features']
        test_labels = test_data['labels']
        test_groups = test_data['groups']
        
        print(f"Test samples: {test_features.shape[0]}")
        print(f"Test queries: {len(test_groups)}")
        
        
        dtest = xgb.DMatrix(test_features)
        predictions = self.model.predict(dtest)
        

        mrr_list = []
        hr_at_1 = []
        hr_at_5 = []
        hr_at_10 = []
        hr_at_15 = []
        hr_at_20 = []
        ndcg_at_5_list = []
        ndcg_at_10_list = []
        
        start_idx = 0
        for group_size in test_groups:
            end_idx = start_idx + group_size
            
            group_scores = predictions[start_idx:end_idx]
            group_labels = test_labels[start_idx:end_idx]
            
            
            sorted_indices = np.argsort(-group_scores)
            sorted_labels = group_labels[sorted_indices]
            
            # MRR
            positive_positions = np.where(sorted_labels == 1)[0]
            if len(positive_positions) > 0:
                first_pos = positive_positions[0] + 1
                mrr_list.append(1.0 / first_pos)
                
                hr_at_1.append(1.0 if first_pos <= 1 else 0.0)
                hr_at_5.append(1.0 if first_pos <= 5 else 0.0)
                hr_at_10.append(1.0 if first_pos <= 10 else 0.0)
                hr_at_15.append(1.0 if first_pos <= 15 else 0.0)
                hr_at_20.append(1.0 if first_pos <= 20 else 0.0)
            
            # NDCG
            if len(group_labels) >= 5:
                ndcg_5 = ndcg_score(
                    group_labels.reshape(1, -1),
                    group_scores.reshape(1, -1),
                    k=5
                )
                ndcg_at_5_list.append(ndcg_5)
            
            if len(group_labels) >= 10:
                ndcg_10 = ndcg_score(
                    group_labels.reshape(1, -1),
                    group_scores.reshape(1, -1),
                    k=10
                )
                ndcg_at_10_list.append(ndcg_10)
            
            start_idx = end_idx
        
        results = {
            'MRR': np.mean(mrr_list),
            'HR@1': np.mean(hr_at_1),
            'HR@5': np.mean(hr_at_5),
            'HR@10': np.mean(hr_at_10),
            'HR@15': np.mean(hr_at_15),
            'HR@20': np.mean(hr_at_20),
            'NDCG@5': np.mean(ndcg_at_5_list) if ndcg_at_5_list else 0.0,
            'NDCG@10': np.mean(ndcg_at_10_list) if ndcg_at_10_list else 0.0
        }
        
        print(f"\nTest Results:")
        print(f"  MRR:      {results['MRR']:.4f}")
        print(f"  HR@1:     {results['HR@1']:.4f}")
        print(f"  HR@5:     {results['HR@5']:.4f}")
        print(f"  HR@10:    {results['HR@10']:.4f}")
        print(f"  HR@15:    {results['HR@15']:.4f}")
        print(f"  HR@20:    {results['HR@20']:.4f}")
        print(f"  NDCG@5:   {results['NDCG@5']:.4f}")
        print(f"  NDCG@10:  {results['NDCG@10']:.4f}")
        

        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved: {results_path}")
        
        return results
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'xgboost_ranker.json')
        
        print(f"Loading model from {model_path}...")
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        feature_path = os.path.join(self.output_dir, 'xgboost_feature_names.pkl')
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
        
        print("Model loaded")
        
        return self.model


def main():
    print("\n" + "="*70)
    print("XGBOOST RANKER TRAINING - COMPLETE PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ranker = XGBoostRanker(output_dir='checkpoints')
    

    try:
        model = ranker.train(
            train_path='xgboost_features/train_features.pkl',
            val_path='xgboost_features/val_features.pkl',
            test_path='xgboost_features/test_features.pkl',
            num_boost_round=500,
            early_stopping_rounds=50
        )
        

        test_results = ranker.evaluate('xgboost_features/test_features.pkl')
        

        print("\n" + "="*70)
        print("RAINING COMPLETE!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nOutput files:")
        print("  - checkpoints/xgboost_ranker.json (model)")
        print("  - checkpoints/training_curves.png (plot)")
        print("  - checkpoints/feature_importance.pkl (importance)")
        print("  - checkpoints/test_results.json (results)")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR OCCURRED")
        print("="*70)
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
        
        return None
    
    return model


if __name__ == "__main__":
    model = main()