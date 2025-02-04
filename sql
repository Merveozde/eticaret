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
