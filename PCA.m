clear
clc

%loading the data
load('AC50001_assignment2_data.mat');

%combinig all data
comb_data = [digit_one digit_five digit_eight]';

%%%%%%%%%%%%%%%%%%PREPARATION%%%%%%%%%%%%%%%%%%%%%%%
[score] = PCA_calc(comb_data);
[idx,c,x1,x2,x_grid,idx_2_region] = PLOT_RESULT(score);
%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. PCA calculation


function [scores] = PCA_calc(comb_data)

meandigits = mean(comb_data);
comb_data = comb_data - repmat(meandigits, size(comb_data,1),1);
covar = cov(comb_data);
[v,d] = eigs(covar);
scores = comb_data*v(:,1:2);

end

% 2. Plots

function [idx,c,x1,x2,x_grid,idx_2_region] = Scatter3Clust(pca_score)
 
[idx, c] = kmeans(pca_score,3);
x1 = min(pca_score(:,1)):0.01:max(pca_score(:,1));
x2 = min(pca_score(:,2)):0.01:max(pca_score(:,2));
[x1G,x2G] = meshgrid(x1,x2);
x_grid = [x1G(:),x2G(:)]; 
idx_2_region = kmeans(x_grid,3,'MaxIter',1,'Start',c);

end



function [idx,c,x1,x2,x_grid,idx_2_region] = PLOT_RESULT(PCA_score)

[idx,c,x1,x2,x_grid,idx_2_region] = Scatter3Clust(PCA_score);
    
figure;
gscatter(x_grid(:,1),x_grid(:,2),idx_2_region,...
    [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'..');
hold on;
plot(PCA_score(:,1),PCA_score(:,2),'k*','MarkerSize',5);
title 'Vectorised digits 1,5,8';
xlabel 'dim1';
ylabel 'dim2';
legend('Ones','Fives','Eights','Data','Location','SouthEast');
hold off;    

opts = statset('Display','final');
[idx, c] = kmeans(PCA_score,3,'Distance','cityblock',...
    'Replicates',5,'Options',opts);

figure;
plot(PCA_score(idx==1,1),PCA_score(idx==1,2),'r.','MarkerSize',10)
hold on
plot(PCA_score(idx==2,1),PCA_score(idx==2,2),'g.','MarkerSize',10)
hold on
plot(PCA_score(idx==3,1),PCA_score(idx==3,2),'b.','MarkerSize',10)

plot(c(:,1),c(:,2),'kx',...
     'MarkerSize',10,'LineWidth',3)
legend('Ones','Fives','Eights','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids in digits 1,5,8'
hold off

figure;

scatter(PCA_score(1:100,1), PCA_score(1:100,2),'Marker','x'); 
hold on;
scatter(PCA_score(101:200,1), PCA_score(101:200,2),'Marker','+');
hold on;
scatter(PCA_score(201:300,1), PCA_score(201:300,2), 'Marker','*');
title 'PCA digits 1,5,8'
legend('Ones','Fives','Eights',...
       'Location','NW')
hold off;
end
