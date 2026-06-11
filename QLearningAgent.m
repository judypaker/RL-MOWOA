classdef QLearningAgent < handle
    %% Q-Learning Agent Class
    % Used for adaptive adjustment of MOWOA algorithm parameters.
    % Learns the optimal parameter combination to improve algorithm performance.
    
    properties
        q_table          % Q-table
        learning_rate    % Learning rate
        discount_factor  % Discount factor
        epsilon          % Exploration rate
        epsilon_decay    % Exploration rate decay
        epsilon_min      % Minimum exploration rate
        state_size       % State space size
        action_size      % Action space size
        actions          % Action set (parameter combinations)
        current_episode  % Current episode number
    end
    
    methods
        function obj = QLearningAgent()
            %% Constructor - Initialize Q-learning Agent
            
            % Hyperparameter settings
            obj.learning_rate = 0.12;
            obj.discount_factor = 0.95;
            obj.epsilon = 0.35;
            obj.epsilon_decay = 0.9975;
            obj.epsilon_min = 0.05;
            obj.current_episode = 0;
            
            % Define state space (discretized)
            % State includes: [convergence, diversity, iteration progress, population quality]
            obj.state_size = 4^4; % 4 discrete values for each state dimension
            
            % Define action space (parameter combinations)
            % Action includes: [scaling factor SF, spiral parameter b, exploration probability p, mutation rate]
            obj.actions = obj.define_action_space();
            obj.action_size = size(obj.actions, 1);
            
            % Initialize Q-table
            obj.q_table = zeros(obj.state_size, obj.action_size);
            
            fprintf('Q-Learning Agent initialization complete\n');
            fprintf('State space size: %d, Action space size: %d\n', obj.state_size, obj.action_size);
        end
        
        function actions = define_action_space(obj)
            %% Define Action Space - different parameter combinations
            % Each action is a parameter vector: [SF, b, p, mutation_rate]
            
            SF_values = [1.15, 1.25, 1.35, 1.45];
            b_values = [1.2, 1.4, 1.6, 1.8];
            p_values = [0.60, 0.65, 0.70, 0.75];
            mutation_values = [0.08, 0.10, 0.12, 0.14];
            
            % Generate all parameter combinations
            [SF_grid, b_grid, p_grid, mut_grid] = ndgrid(SF_values, b_values, p_values, mutation_values);
            
            actions = [SF_grid(:), b_grid(:), p_grid(:), mut_grid(:)];
        end
        
        function state_index = discretize_state(obj, state_vector)
            %% Discretize continuous state vector into state index
            % state_vector: [convergence, diversity, iteration progress, population quality]
            
            % Discretize each state dimension into 4 intervals
            discrete_state = zeros(1, 4);
            
            for i = 1:4
                if state_vector(i) <= 0.25
                    discrete_state(i) = 1;
                elseif state_vector(i) <= 0.5
                    discrete_state(i) = 2;
                elseif state_vector(i) <= 0.75
                    discrete_state(i) = 3;
                else
                    discrete_state(i) = 4;
                end
            end
            
            % Convert multi-dimensional discrete state to one-dimensional index
            state_index = sub2ind([4, 4, 4, 4], discrete_state(1), discrete_state(2), ...
                                 discrete_state(3), discrete_state(4));
        end
        
        function [action_index, params] = select_action(obj, state_vector)
            %% Select Action - Using epsilon-greedy strategy
            
            state_index = obj.discretize_state(state_vector);
            
            if rand() < obj.epsilon
                % Exploration: random action
                action_index = randi(obj.action_size);
            else
                % Exploitation: select action with maximum Q-value
                [~, action_index] = max(obj.q_table(state_index, :));
            end
            
            % Return corresponding parameter combination
            params = obj.actions(action_index, :);
        end
        
        function update_q_table(obj, state_vector, action_index, reward, next_state_vector)
            %% Update Q-table - Using Q-learning update rule
            
            state_index = obj.discretize_state(state_vector);
            next_state_index = obj.discretize_state(next_state_vector);
            
            % Q-learning update formula
            current_q = obj.q_table(state_index, action_index);
            max_next_q = max(obj.q_table(next_state_index, :));
            
            new_q = current_q + obj.learning_rate * (reward + obj.discount_factor * max_next_q - current_q);
            obj.q_table(state_index, action_index) = new_q;
            
            % Update exploration rate
            if obj.epsilon > obj.epsilon_min
                obj.epsilon = obj.epsilon * obj.epsilon_decay;
            end
            
            obj.current_episode = obj.current_episode + 1;
        end
        
        function save_agent(obj, filename)
            %% Save trained agent
            save(filename, 'obj');
            fprintf('Q-Learning agent saved to: %s\n', filename);
        end
        
        function load_agent(obj, filename)
            %% Load pre-trained agent
            if exist(filename, 'file')
                loaded_data = load(filename);
                obj.q_table = loaded_data.obj.q_table;
                obj.epsilon = loaded_data.obj.epsilon;
                obj.current_episode = loaded_data.obj.current_episode;
                fprintf('Q-Learning agent loaded from file: %s\n', filename);
            else
                fprintf('Warning: File %s does not exist, using default initialization\n', filename);
            end
        end
        
        function visualize_q_table(obj)
            %% Visualize Q-table - Display learned policy
            
            figure('Name', 'Q-Table Visualization', 'Position', [100, 100, 400, 300]);
            
            % Display optimal action for each state
            [~, optimal_actions] = max(obj.q_table, [], 2);
            plot(optimal_actions, 'o-', 'LineWidth', 2);
            title('Optimal Action for Each State');
            xlabel('State Index');
            ylabel('Optimal Action Index');
            grid on;
        end
        
        function stats = get_learning_stats(obj)
            %% Get learning statistics
            
            stats = struct();
            stats.total_episodes = obj.current_episode;
            stats.current_epsilon = obj.epsilon;
            stats.q_table_mean = mean(obj.q_table(:));
            stats.q_table_std = std(obj.q_table(:));
            stats.q_table_max = max(obj.q_table(:));
            stats.q_table_min = min(obj.q_table(:));
            
            % Calculate how many times each action is selected as optimal
            [~, optimal_actions] = max(obj.q_table, [], 2);
            stats.action_preferences = histcounts(optimal_actions, 1:obj.action_size+1);
            
            fprintf('=== Q-Learning Agent Learning Statistics ===\n');
            fprintf('Total Episodes: %d\n', stats.total_episodes);
            fprintf('Current Epsilon: %.4f\n', stats.current_epsilon);
            fprintf('Q-table Mean: %.4f\n', stats.q_table_mean);
            fprintf('Q-table Std: %.4f\n', stats.q_table_std);
            fprintf('Q-table Max: %.4f\n', stats.q_table_max);
            fprintf('Q-table Min: %.4f\n', stats.q_table_min);
        end
    end
end
