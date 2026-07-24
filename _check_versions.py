import shap
print('shap_version=', shap.__version__)
try:
    import xgboost
    print('xgboost_version=', xgboost.__version__)
except Exception as exc:
    print('xgboost_import_error=', exc)
