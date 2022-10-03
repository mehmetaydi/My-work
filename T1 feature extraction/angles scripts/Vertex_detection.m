jpegFiles = dir('*.png');
numfiles = length(jpegFiles);
% numfiles = 10; %500;
Vertex = cell(1, numfiles);
% mydata = cell(1, numfiles);

fid = fopen('single_predictions.txt', 'rt');
C = textscan(fid, '%f%f%f%f%f%f', 'MultipleDelimsAsOne', true, 'Delimiter', ' ');
fclose(fid);
frame_col = cell2mat(C(3));
x_col = cell2mat(C(5));
y_col = cell2mat(C(4));
tic
% ------------------------------------------------------------------------
for frame_idx = 1:numfiles 

  FFC = imread(jpegFiles(frame_idx).name); 
  
  % [FF, ~] = imcrop(FFC,[0 600 900 100]);   
  % [FF, ~] = imcrop(FFC, [0 0 680 680]); 
  
%  F = im2bw(FFC); %imbinarize(F);
%  FR = imfill(F,8,'holes'); 
  % mydata2{1, k} = FR;
  
  
  % FR = imfill(im2bw(FFC),8,'holes');
  %FR = im2bw(FFC);
  FR = im2bw(~FFC);
  
  [row, col] = size(FR); 

  FR2 = zeros(row,col);
  
%% First filter

  for q = 2:col-1

    for r = 2:row-1
    
            if FR(r,q) == 0 
        
            Vx = [FR(r-1,q)   ; FR(r+1,q)   ; FR(r,q-1)   ; FR(r,q+1)   ; ...
                  FR(r-1,q-1) ; FR(r+1,q-1) ; FR(r-1,q+1) ; FR(r+1,q+1)];
               
                if numel(Vx(Vx == 0)) == 3
              
                    FR2(r,q) = 1;
                    
                end
                
            end   
            
    end
    
  end

%% Find vertex

  inde = find(FR2 == 1); 

  for ii = 1:size(inde,1)

    [iy,ix] = ind2sub(size(FR2),inde(ii));

    V(ii,1) = iy;
    V(ii,2) = ix;

  end

  pdispixel = zeros(size(inde,1),1);

  for ii = 1:size(inde,1)-1

    pdispixel(ii,1) = pdist2(V(ii,:),V(ii+1,:));

  end

  ii = 1;

  while ii < size(inde,1)
    
    if pdispixel(ii,1) > 2
        
        VV(ii,:) = V(ii,:);
        
    else
        
        VV(ii,1:2) = 0;
        
    end
    
    ii = ii + 1;
    
  end

  VV(VV(:,1)==0,:) = [];

%% Create a image of the vertices

% FRR = zeros(size(FR));

  [rows, columns] = size(FR);

  circleImage = false(rows, columns); 
  [x, y] = meshgrid(1:columns, 1:rows); 

  for ii = 1:size(VV,1)
    
    circleImage((x - VV(ii,2)).^2 + (y - VV(ii,1)).^2 <= 2^2) = true; 

  end

  s = regionprops(circleImage,'Centroid');
  centroidsvertex = cat(1,s.Centroid); 
  
  
  
  
  Vertex{1, frame_idx} = centroidsvertex; %Positions of bubble vertices 
  cell_to_mat = cell2mat(Vertex(1,frame_idx));
  
  imshow(jpegFiles.name)
  hold on
  plot(cell_to_mat(:,1), cell_to_mat(:,2),'o')
  
  
  
  
  newStr = split(jpegFiles(1,1).name, ["im", ".png"]);
  FrameNum = str2double(newStr(2,1));
  
  frame_idx = find(frame_col(:,1)==FrameNum);
  x_coord = x_col(frame_idx,1);
  y_coord = y_col(frame_idx,1);
  
  t1_mat = [x_coord; y_coord];
  
  vertex_matrix = cell2mat(Vertex(1));
  distance_vector = [];
  for i = 1:max(size(vertex_matrix))
       current_vertex = vertex_matrix(i,:);
       distance = norm(t1_mat - current_vertex.');
       distance1 = abs(t1_mat(1)-current_vertex(1));
       distance2 = abs(t1_mat(2)-current_vertex(2));
       if distance < 65
           distance_vector = [distance_vector; distance vertex_matrix(i,:)];
           %plot(current_vertex(1), current_vertex(2),'o', 'Color', 'green')
       end
  end
  temp_suspect_t1=1;
  suspect_t1 = [];
  for i = 1:max(size(distance_vector))
       current_vertex = [distance_vector(i,2); distance_vector(i,3)];
       distance = norm(t1_mat - current_vertex);
       distance1 = abs(t1_mat(1)-current_vertex(1));
       distance2 = abs(t1_mat(2)-current_vertex(2));
       if distance < 1
           temp_suspect_t1 = distance_vector(i,:);
           suspect_t1 = [suspect_t1; temp_suspect_t1];
           t1_mat = [current_vertex(1); current_vertex(2)];
       end
  end
  unique_suspects = unique(suspect_t1, 'rows', 'stable');
  %setdiff(unique_suspects, distance_vector,'rows');
  Lia = ismember(distance_vector, unique_suspects, 'rows');
  distance_vector = distance_vector(~Lia,:);
  
  
  suspected_vertices = [];
  for j = 1:max(size(distance_vector))
      vertex_info1 = distance_vector(j,:);
      vertex_vector1 = [vertex_info1(2); vertex_info1(3)];
      for k = 1:max(size(distance_vector))
          vertex_info2 = distance_vector(k,:);
          vertex_vector2 = [vertex_info2(2); vertex_info2(3)];
          temp_dist = norm(vertex_vector1 - vertex_vector2);
          if temp_dist < 30 & temp_dist ~= 0
            dist_to_T1_1 = norm(t1_mat - vertex_vector1);
            dist_to_T1_2 = norm(t1_mat - vertex_vector2);
            if dist_to_T1_1 < dist_to_T1_2
                suspected_vertices = [suspected_vertices; vertex_info2];
            else
                suspected_vertices = [suspected_vertices; vertex_info1];
            end
          end
      end
  end
  
  unique_suspects = unique(suspected_vertices, 'rows', 'stable');
  %setdiff(unique_suspects, distance_vector,'rows');
  Lia = ismember(distance_vector, unique_suspects, 'rows');
  distance_vector_new = distance_vector(~Lia,:);
  %plot(distance_vector_new(:,2), distance_vector_new(:,3),'o','Color','yellow')
  dvna = [];
  for i = 1:max(size(distance_vector_new))
     center = [t1_mat(1), t1_mat(2)];
     current_vertex = [distance_vector_new(i, 2), distance_vector_new(i, 3)];
     angle_round_axis = (atan2(((current_vertex(1))-(center(1))),(current_vertex(2)-center(2))));
     if angle_round_axis < 0
         angle_round_axis = angle_round_axis;
     end
     angle_round_axis = angle_round_axis*(180/pi);
     dvna = [dvna; distance_vector_new(i,:) angle_round_axis];
     
  end
  dvna = sortrows(dvna, 4,'ascend');
  distance_vector_new = dvna;
  suspects = [];
  for j = 1:max(size(distance_vector_new))
     forward_pass=0;
     backward_pass=0;
     P0 = [t1_mat(1), t1_mat(2)];
     if j~= max(size(distance_vector_new))
         P1 = [distance_vector_new(j, 2), distance_vector_new(j, 3)];
         P2 = [distance_vector_new(j+1, 2), distance_vector_new(j+1, 3)];
     else
         P1 = [distance_vector_new(j, 2), distance_vector_new(j, 3)];
         P2 = [distance_vector_new(1, 2), distance_vector_new(1, 3)];
     end
     n1 = (P2 - P0) / norm(P2-P0);
     n2 = (P1 - P0) / norm(P1-P0);
     temp_angle = acos(dot(n1, n2)) * (180/pi);
     if temp_angle < 50
         for i = 1:max(size(distance_vector_new))
            if i~= max(size(distance_vector_new))
                P2_2 = [distance_vector_new(i+1, 2), distance_vector_new(i+1, 3)];
            else
                P2_2 = [distance_vector_new(1, 2), distance_vector_new(1, 3)];
            end
            n1_2 = (P2_2 - P0) / norm(P2_2-P0);
            
            temp_angle_1 = acos(dot(n1_2, n2)) * (180/pi);
            if temp_angle_1 < 50 && n1_2(1) ~= n1(1) && n1_2(2) ~= n1(2)
                forward_pass = 1;
              
                if j == max(size(distance_vector_new))
                    suspect = distance_vector_new(1,:);
                else
                    suspect = distance_vector_new(i+1,:);
                end
                suspects = [suspects; suspect];
            end
         end
     end
     
     
     
   
     
     
     
     
  end
  unique_suspects_2 = unique(suspects, 'rows', 'stable');
  Lia = ismember(distance_vector_new, unique_suspects_2, 'rows');
  distance_vector_new_2 = distance_vector_new(~Lia,:);
  plot(distance_vector_new_2(:,2), distance_vector_new_2(:,3),'o','Color','red')
  % 114.47, 80.63, 105.45, 59.45
  %atan2d(vec1(1)*vec2(2)-vec2(1)*vec1(2), vec1(1)*vec1(2)+vec2(1)*vec2(2));
  
  fileID = fopen('t1_loc.txt', 'w');
  formatSpec = '%4.2f %4.2f %4.2f A0 A1 A2 A3 A4 1';
  fprintf(fileID, formatSpec, FrameNum, x_coord, y_coord);
  fclose(fileID);
  
  
  
  
  
 
%% Clear variables
  
  clear FFC row col q FR2 r Vx VV ii ix iy ...
        rows columns circleImage x y centroidsvertex s ...
        k pdispixel V FR inde

end
toc