import mlflow
import dagshub
 
 
mlflow.set_tracking_uri('https://dagshub.com/ruturaj0626/Auto_MPG.mlflow')

dagshub.init(repo_owner='ruturaj0626', repo_name='Auto_MPG', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)