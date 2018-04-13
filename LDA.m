%loading data
load('AC50001_assignment2_data.mat');

%combining all data
comb_data = [digit_one digit_five digit_eight]' ;
data_label = [];

meandigits = mean(comb_data);
comb_data = comb_data - repmat(meandigits, size(comb_data,1),1);


%getting labels 
for k=1:size(comb_data,1)
    if k<= 100
        data_label = [data_label;'1'];
    end
    if k >100 && k <= 200
        
        data_label = [data_label;'5'];
    end
    if k > 200
        data_label = [data_label;'8'];
    end
end



classOnes = comb_data(data_label=='1',:);
classFives = comb_data(data_label=='5',:);
classEights = comb_data(data_label=='8',:);

%means
mOnes = mean(classOnes,1);
mFives = mean(classFives,1);
mEights = mean(classEights,1);

%class covariance matrices 
covarOnes = cov(classOnes);
covarFives = cov(classFives);
covarEights = cov(classEights);

%within class scatter matrix
sw = covarOnes + covarFives + covarEights;

%mean of the class means
meanClassMeans = (mOnes + mFives + mEights)./3;

%each of classes has 100 samples, no need to
%recalculate using size(class,2)

%between class scatter matrix
sbOnes = 100 .* (mOnes-meanClassMeans)'*(mOnes-meanClassMeans);
sbFives = 100 .* (mFives-meanClassMeans)'*(mFives-meanClassMeans);
sbEights = 100 .* (mEights-meanClassMeans)'*(mEights-meanClassMeans);

sb = sbOnes+ sbFives+sbEights;

%LDA projection vector

%addding small number to avoid inv on zero
dc=0.00001*eye(size(sw));

sw_new=sw+dc;

inverted_SW=inv(sw_new)*sb;
    
% computing the projection vectors:
[v1,d] = eig(inverted_SW);

% calculating score
scores = (comb_data*v1(:,1:2));

%Plot
figure


scatter(scores(1:100,1), scores(1:100,2), 'r', 'x'); %% look at only one direction will be fine
hold on;
scatter(scores(101:200,1), scores(101:200,2),'b', '*');
hold on;
scatter(scores(201:300,1), scores(201:300,2), 'g', 'o');
title 'LDA';
legend('Ones','Fives','Eights','Data','Location','NorthEast');  
hold off;