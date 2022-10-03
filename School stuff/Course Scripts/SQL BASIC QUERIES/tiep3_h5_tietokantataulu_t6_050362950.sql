select distinct artist.artist_id, first_name ,last_name FROM artist ,artwork
where exists 
(select * from artwork
where artwork.artist_id = artist.artist_id and
(technique = 'painting'  ) and (
artwork.artist_id = '2' or artwork.artist_id = '3' or 
artwork.artist_id = '4')  )
ORDER BY artist.artist_id ASC;

-- Tietokantojen perusteet 2020
-- H6 T6
-- mehmet.aydin@tuni.fi 