select distinct artwork_name, value ,year_created FROM artwork

where value =
(select max(value) from  artwork 
);

-- Tietokantojen perusteet 2020
-- H6 T7
-- mehmet.aydin@tuni.fi 