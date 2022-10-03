SELECT distinct gender,byear ,uname FROM user ,brand,evaluation,product

where evaluation.u_id = users.users_id and evaluation.p_id = product.p_id

and product.b_id = brand.b_id  and bname = 'McCee' 

INTERSECT


SELECT distinct gender,byear ,uname FROM user ,brand,evaluation,product

where evaluation.u_id = users.users_id and evaluation.p_id = product.p_id

and product.b_id = brand.b_id  and bname = 'KooTek' 
