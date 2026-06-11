function f  = replace_chromosome(intermediate_chromosome, M,D,NP)

%% function f  = replace_chromosome(intermediate_chromosome,M,D,NP)
% This function replaces the chromosomes based on rank and crowding
% distance. Initially until the population size is reached each front is
% added one by one until addition of a complete front which results in
% exceeding the population size. At this point the chromosomes in that
% front is added subsequently to the population based on crowding distance.

[~,m]=size(intermediate_chromosome);
f=zeros(NP,m);

sorted_chromosome = sortrows(intermediate_chromosome,M + D + 1);

max_rank = max(intermediate_chromosome(:,M + D + 1));

previous_index = 0;
for i = 1 : max_rank
    current_index = find(sorted_chromosome(:,M + D + 1) == i, 1, 'last' );
    if current_index > NP
        remaining = NP - previous_index;
        temp_pop = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
        [~,temp_sort_index] = ...
            sort(temp_pop(:, M + D + 2),'descend');
        for j = 1 : remaining
            f(previous_index + j,:) = temp_pop(temp_sort_index(j),:);
        end
        return;
    elseif current_index < NP
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
    else
        f(previous_index + 1 : current_index, :) = ...
            sorted_chromosome(previous_index + 1 : current_index, :);
        return;
    end
    previous_index = current_index;
end
