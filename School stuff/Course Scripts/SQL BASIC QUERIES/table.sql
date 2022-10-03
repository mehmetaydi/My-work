
CREATE TABLE users (
    u_id INT NOT NULL AUTO_INCREMENT,
    uname VARCHAR (40) NOT NULL,
    gender VARCHAR (20) NOT NULL,
    byear INT,	
    PRIMARY KEY (u_id),
	UNIQUE(uname)
);
INSERT INTO users

VALUES('Anni N','female', '1985');

INSERT INTO users

VALUES('Juuso K','male', '1990');

INSERT INTO users

VALUES('Eino U','male', '1940');

INSERT INTO users

VALUES('Aila J','female', '1953');

CREATE TABLE brand (
    b_id INT NOT NULL AUTO_INCREMENT,
    bname VARCHAR (40) NOT NULL,
    country VARCHAR (20),
    PRIMARY KEY (b_id),
    UNIQUE(bname)
);

INSERT INTO brand

VALUES('McCee','United States ');

INSERT INTO brand

VALUES('KooTek','Finland');

CREATE TABLE category (
    c_id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR (40) NOT NULL,
    PRIMARY KEY (c_id),
    UNIQUE(cname)
);

INSERT INTO category

VALUES('garden');

INSERT INTO category

VALUES('forest');


INSERT INTO category

VALUES('snow clearing ');


CREATE TABLE product (
    p_id INT NOT NULL AUTO_INCREMENT,
    pname VARCHAR (40) NOT NULL,
    description VARCHAR (40) NOT NULL,
    price INT,	
	b_id INT NOT NULL ,
	UNIQUE(pname),
    PRIMARY KEY (p_id),
	FOREIGN KEY (b_id) REFERENCES brand(b_id)
);

INSERT INTO product

VALUES('Grass trimmer TRCee','efficient 4-stroke', ' 179.00 ','1');

INSERT INTO product

VALUES('Trimmer line Cee','high-class line', ' 6.99 ','1');

INSERT INTO product

VALUES('Chain saw MSCee RR','robust and heavy', '  559.00  ','1');

INSERT INTO product

VALUES('Trimmer line Y,','all-purpose line', '  3.99  ','2');

INSERT INTO product

VALUES('Shovel L',' light general-purpose shovel', '23.95','2');



CREATE TABLE evaluation (
    u_id INT NOT NULL,
    p_id INT NOT NULL,
	edate DATE,
	rating INT,
	vreview VARCHAR (40),
    PRIMARY KEY (u_id, p_id,edate),
    FOREIGN KEY (u_id) REFERENCES users(u_id),
    FOREIGN KEY (p_id) REFERENCES product(p_id)
	
);

INSERT INTO evaluation
VALUES('1',' 1', '2017-06-05','3','');

INSERT INTO evaluation
VALUES('1',' 2', '2017-06-13','2','');

INSERT INTO evaluation
VALUES('1',' 5', '2017-07-24 ','3','');

INSERT INTO evaluation
VALUES('1',' 4', '2017-08-13 ','4','');


INSERT INTO evaluation
VALUES('1',' 1', '2017-09-12 ','5', 'reliable and functioning gadget');

INSERT INTO evaluation
VALUES('3',' 1', '2017-06-30  ','5', 'excellent');

INSERT INTO evaluation
VALUES('3',' 2', '2017-07-02  ','2', 'moderately works');

INSERT INTO evaluation
VALUES('2',' 5', '2017-06-04 ','1', 'rip-off');

INSERT INTO evaluation
VALUES('4',' 5', '2017-08-11  ','1', 'completely useless');



CREATE TABLE product_category (
    c_id INT NOT NULL ,
    p_id INT NOT NULL ,
    PRIMARY KEY (c_id, p_id),
    FOREIGN KEY (c_id) REFERENCES category(c_id),
    FOREIGN KEY (p_id) REFERENCES product(p_id)

);

INSERT INTO product_category

VALUES('1','1');

INSERT INTO product_category

VALUES('2','1');

INSERT INTO product_category

VALUES('3','1');

INSERT INTO product_category

VALUES('3','2');

INSERT INTO product_category

VALUES('4','1');

INSERT INTO product_category

VALUES('5','1');

INSERT INTO product_category

VALUES('5','3');