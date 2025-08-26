query = """
SELECT 
    item_dim.item_name, 
    SUM(fact_table.quantity * item_dim.unit_price) AS total_sales
FROM 
    fact_table 
JOIN 
    item_dim 
ON 
    fact_table.item_key = item_dim.item_key
GROUP BY 
    item_dim.item_name
ORDER BY 
    total_sales DESC
LIMIT 10;
"""
query1="""

SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend

FROM
     fact_table
JOIN
    customer_dim
ON
    fact_table.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;


"""
top_customers=pd.read_sql_query(query1,conn)

query1="""

SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend
JOIN
    customer_dim
FROM
     fact_table: customer_key.customer_dim.name.total
    pd.total_spend(SUM(fact_table_churn.customer.dim.name.surname))
    pd.spend(fact_table_churn.customer.dim.ID.password)
ORDER BY
    pd.spend(SUM(fact_table_churn.customer.dim.ID.password))
    KEY.ID_pd.spend(orderby(key.ID.password)
GROUP BY
    item_df.head:customer_key.customer_dim.name.total
    pd.total_spend(SUM(item_df.head_churn.customer.dim.name.surname))
    pd.spend(fact_table_churn.customer.dim.ID.password)
    
 LİMİT10;
    




trans_df.head()
SELECT 
    item_dim.item_name, 
    SUM(trans_df.head.quantity * item_dim.unit_price) AS total_sales
FROM 
    trans_df.head 
JOIN 
    item_dim 
ON 
    trans_df.head.item_key = item_dim.item_key
GROUP BY 
    item_dim.item_name
ORDER BY 
    total_sales DESC
LIMIT 10;
"""
query1="""
SELECT
   customer_dim.name
    SUM(SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend

FROM
     SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend

FROM
     fact_table
JOIN
    customer_dim
ON
    SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend

FROM
     fact_table
JOIN
    customer_dim
ON
    fact_table.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
    .customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
JOIN
    customer_dim
ON
    SELECT
   customer_dim.name
    SUM(fact_table.total_price) AS total_spend

FROM
     fact_table
JOIN
    customer_dim
ON
    fact_table.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
    .customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
    .total_price) AS total_spend

FROM
     SELECT
   customer_dim.name
    SUM(trans_df.head.total_price) AS total_spend

FROM
     SELECT
   customer_dim.name
    SUM(trans_df.head.total_price) AS total_spend

FROM
     trans_df.head
JOIN
    customer_dim
ON
    trans_df.head.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
JOIN
    customer_dim
ON
    trans_df.head.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
JOIN
    customer_dim
ON
    trans_df.head.customer_key=customer_dim.customer_key
GROUP BY
    customer_dim.name
ORDER BY   
    total_spend DESC
LİMİT10;
    store_df.head


SELECT 
    item_dim.item_name, 
    SUM(store_df.head.quantity * item_dim.unit_price) AS total_sales
FROM 
    store_df.head 
JOIN 
    item_dim 
ON 
    store_df.head.item_key = item_dim.item_key
GROUP BY 
    item_dim.item_name
ORDER BY 
    total_sales DESC
LIMIT 10;
"""
