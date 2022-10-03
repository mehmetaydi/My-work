SELECT uname FROM user ,brand,evaluation,product
where evaluation.u_id = users.u_id and evaluation.p_id = product.p_id
and product.b_id = brand.b_id  and  (pname ='Trimmer line Y,' or 'Shovel L')


---SELECT users.name FROM users ,brand,evaluation,product
---where evaluation.users_id = users.users_id and evaluation.product_id = product.product_id
---and product.brand_id = brand.brand_id  and  (product.name ='Trimmer line Y,' or 'Shovel L')