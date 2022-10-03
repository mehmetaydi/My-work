SELECT distinct bname,pname ,date,rating FROM brand,product,evaluation,user

where product.p_id = evaluation.p_id and brand.b_id = product.b_id and user.u_id = evaluation.u_id
and uname = 'Anni N'

ORDER BY bname ASC, pname  ASC,date DESC;