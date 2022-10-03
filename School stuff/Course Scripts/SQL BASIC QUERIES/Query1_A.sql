SELECT distinct bname,pname ,date,rating FROM brand,product,evaluation

where product.p_id = evaluation.p_id and brand.b_id = product.b_id

ORDER BY bname ASC, pname  ASC,date DESC;