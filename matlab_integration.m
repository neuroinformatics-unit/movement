% MATLAB Script to Load and Analyze Movement Dataset
clc;
clear all; 
close all;

filename = 'movement_data.csv';
data = readtable(filename);

disp('First few rows of the dataset:');
head(data)

if iscell(data.Time) || ischar(data.Time)
    data.Time = str2double(data.Time);
end

figure;
subplot(2,1,1);
plot(data.X_Position, data.Y_Position, '-o');
title('Movement Trajectory');
xlabel('X Position'); ylabel('Y Position');
grid on;

meanX = mean(data.X_Position);
meanY = mean(data.Y_Position);
stdX = std(data.X_Position);
stdY = std(data.Y_Position);

fprintf('Mean X Position: %.2f\n', meanX);
fprintf('Mean Y Position: %.2f\n', meanY);
fprintf('Standard Deviation of X Position: %.2f\n', stdX);
fprintf('Standard Deviation of Y Position: %.2f\n', stdY);

timeDiff = diff(data.Time);
distanceDiff = sqrt(diff(data.X_Position).^2 + diff(data.Y_Position).^2); 
speed = distanceDiff ./ (timeDiff + eps); 

meanSpeed = mean(speed);
fprintf('Mean Speed: %.2f units/time\n', meanSpeed);

subplot(2,1,2);
plot(data.Time(2:end), speed, '-o'); 
title('Speed Over Time');
xlabel('Time'); ylabel('Speed (units/time)');
grid on;
