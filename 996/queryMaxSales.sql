SELECT 
    t1.legalEntity,
    t1.userId,
    MAX(t1.salesAmount) as max_sales_amount
FROM your_table_name t1
INNER JOIN (
    SELECT 
        legalEntity, 
        userId, 
        MAX(recordeDate) as latest_date
    FROM your_table_name
    GROUP BY legalEntity, userId
) t2 ON t1.legalEntity = t2.legalEntity 
      AND t1.userId = t2.userId 
      AND t1.recordeDate = t2.latest_date
GROUP BY t1.legalEntity, t1.userId;
