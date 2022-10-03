select distinct technique FROM artwork , artist

where (value >
(select avg(value) from  artwork ));

-- Tietokantojen perusteet 2020
-- H6 T9
-- mehmet.aydin@tuni.fi 