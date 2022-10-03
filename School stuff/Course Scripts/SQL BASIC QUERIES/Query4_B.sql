SELECT pname, min(evaluation.rating) as min_rating , max(evaluation.rating) as max_rating , 

avg(evaluation.rating) as avg_rating ,count(evaluation.rating) as no_rating , count(DISTINCT evaluation.u_id) as different_users from product 

left OUTER JOIN  evaluation on   evaluation.p_id =product.p_id 


group by pname