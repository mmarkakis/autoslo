# AutoSLO
Mock extended benchmark workloads by composing chunks



# Workgroup creation runbook

1. Create workgroup
    - Impose base RPU = max RPU
    - Make sure to add the admin password manually.
    - Attach IAM roles (command access, loader)
    - Export logs as needed.
    
    

2. Make publicly accessible:
```bash
 aws redshift-serverless update-workgroup   --workgroup-name <workgroup_name>  --publicly-accessible 
```

3. Attach TPC-DS data
    - Go to query editor v2, connect to dev on the new workgroup.
    - Run:
```SQL

CREATE DATABASE tpcds_db
FROM DATASHARE tpcds_datashare
OF ACCOUNT '147854383891'
NAMESPACE '1015d398-b04c-40d0-bb67-257e0956c96d';

CREATE EXTERNAL SCHEMA ext_tpcds1 FROM redshift DATABASE tpcds_db SCHEMA tpcds1;
CREATE EXTERNAL SCHEMA ext_tpcds10 FROM redshift DATABASE tpcds_db SCHEMA tpcds10;
CREATE EXTERNAL SCHEMA ext_tpcds100 FROM redshift DATABASE tpcds_db SCHEMA tpcds100;
CREATE EXTERNAL SCHEMA ext_tpcds1000 FROM redshift DATABASE tpcds_db SCHEMA tpcds1000;
CREATE EXTERNAL SCHEMA ext_tpcds3000 FROM redshift DATABASE tpcds_db SCHEMA tpcds3000;
CREATE EXTERNAL SCHEMA ext_tpcds10000 FROM redshift DATABASE tpcds_db SCHEMA tpcds10000;

```


