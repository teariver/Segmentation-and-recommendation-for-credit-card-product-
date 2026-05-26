# Customer Distribution by Segment
SELECT 
    SEGMENT,
    COUNT(DISTINCT `ï»¿CUS_ID`) AS total_customers
FROM abc.credit_use_customer_segmented
GROUP BY SEGMENT
ORDER BY total_customers DESC;

# 2 Average Spending Behavior by Segment
# Compare average spending value, credit balance, and card limit across customer groups.

SELECT
    SEGMENT,
    AVG(SHOPPING_VALUE) AS avg_shopping_value,
    AVG(CREDIT_BALANCE) AS avg_credit_balance,
    AVG(CARD_LIMIT) AS avg_card_limit
FROM  abc.credit_use_customer_segmented
GROUP BY SEGMENT
ORDER BY avg_shopping_value DESC;

# 3. Payment Behavior Analysis

# Analyze payment ratio and repayment behavior for each customer segment.

SELECT
    SEGMENT,
    AVG(PAYMENT_RATIO) AS avg_payment_ratio,
    AVG(TOTAL_PAYMENT) AS avg_total_payment,
    AVG(MIN_PAYMENT_AMOUNT) AS avg_min_payment
FROM  abc.credit_use_customer_segmented
GROUP BY SEGMENT
ORDER BY avg_payment_ratio DESC;

#4 Top High-Spending Customers
# Identify customers with the highest spending values.

SELECT
    ï»¿CUS_ID as Cus_ID,
    SEGMENT,
    SHOPPING_VALUE,
    CREDIT_BALANCE,
    CARD_LIMIT,
    PAYMENT_RATIO
FROM  abc.credit_use_customer_segmented
ORDER BY SHOPPING_VALUE DESC
LIMIT 10;

# 5 Branch Performance Analysis

# Analyze customer distribution and spending behavior across bank branches.
SELECT
    BRN_DIM_ID,
    SEGMENT,
    COUNT(DISTINCT `ï»¿CUS_ID` ) AS total_customers,
    AVG(SHOPPING_VALUE) AS avg_shopping_value
FROM CREDIT_USE_CUSTOMER_SEGMENTED
GROUP BY BRN_DIM_ID, SEGMENT
ORDER BY BRN_DIM_ID, total_customers DESC;

# 6. Most Recommended Campaigns

# Identify the most frequently recommended promotional campaigns.

SELECT
    CMPN_DIM_ID,
    COUNT(*) AS recommendation_count
FROM FACT_RECOMMENDATION_PROGRAMS
GROUP BY CMPN_DIM_ID
ORDER BY recommendation_count DESC
LIMIT 10;