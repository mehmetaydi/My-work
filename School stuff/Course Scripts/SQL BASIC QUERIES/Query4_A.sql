SELECT pname, min(evaluation.rating) as min_rating , max(evaluation.rating) as max_rating , 

avg(evaluation.rating) as avg_rating ,count(evaluation.rating) as no_rating FROM product 

left OUTER JOIN  evaluation on   evaluation.p_id =product.p_id 

group by pname