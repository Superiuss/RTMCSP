% Units: s,L,mmol
k1 = 1.39e-6; 
k2 = 8.3e-6;
k3 = 1.24e-3;
k4 = 3.4;
k5 = 0.2115e-3;
k6 = 0.05823;
k71 = 10;
k72 = 1.5e-3;
k8 = 1.75e-6;
k9 = 7.8e-10;
k10 = 3.5e-7;
kdo = 0.165;
kea = 0.11;
kn = 0.01;
kI = 1.6e-5;

% Initial concentrations
o2 = 0.25; h2o2 = 0; fe2 = 0.02; Sr = 5.13; cr6 = 0.038; acr6 = 0; cr3 = 0; no3 = 0.3;
y0 = [o2, h2o2, fe2, Sr, cr6, acr6, cr3, no3];
nSpecies = 8;

% time span
tspan = [0 3600];

% ODEs
[t, y] = ode45(@(t, y) reaction_system(t, y, k1, k2, k3, k4, k5, k6, k71, k72, k8, k9, k10, kdo, kea, kn, kI), tspan, y0);

% Key parts
num_fast_species = 3;
contributions = zeros(length(t),  nSpecies);

a = zeros(nSpecies, nSpecies, length(t));
b = zeros(nSpecies, nSpecies, length(t));

for i = 1:length(t)
    J_numeric = evaluate_jacobian(y(i, :), k1, k2, k3, k4, k5, k6, k71, k72, k8, k9, k10, kdo, kea, kn, kI);
    [V, D] = eig(J_numeric);
    eigenvalues = diag(D);
    [~, idx] = sort(abs(eigenvalues), 'descend');
    V_fast = V(:, idx(1:num_fast_species));
    a(:,:,i) = V;
    b(:,:,i) = inv(V)';
    A_f = a(:, 1:num_fast_species, i);
    B_f = b(1:num_fast_species, :, i);
    for j = 1:size(V_fast, 1)
        contributions(i, j) = max(abs(V_fast(j, :)));
    end
end

% Plot
figure;
plot(t, y(:,1), 'r', 'DisplayName', 'o2');
hold on;
plot(t, y(:,2), 'b', 'DisplayName', 'h2o2');
plot(t, y(:,3), 'g', 'DisplayName', 'fe2');
plot(t, y(:,4), 'k', 'DisplayName', 'Sr');
plot(t, y(:,5), 'c', 'DisplayName', 'cr6');
plot(t, y(:,6), 'y', 'DisplayName', 'acr6');
plot(t, y(:,7), 'm', 'DisplayName', 'cr3');
plot(t, y(:,8), 'Color', '#A2142F', 'DisplayName', 'no3');
xlabel('Time');
ylabel('Concentration');
legend show;
title('Concentration of each Species');
hold off;

figure;
plot(t, contributions(:, 1), 'r', 'DisplayName', 'o2');
hold on;
plot(t, contributions(:, 2), 'b', 'DisplayName', 'h2o2');
plot(t, contributions(:, 3), 'g', 'DisplayName', 'fe2');
plot(t, contributions(:, 4), 'k', 'DisplayName', 'Sr');
plot(t, contributions(:, 5), 'c', 'DisplayName', 'cr6');
plot(t, contributions(:, 6), 'y', 'DisplayName', 'acr6');
plot(t, contributions(:, 7), 'm', 'DisplayName', 'cr3');
plot(t, contributions(:, 8), 'Color', '#A2142F', 'DisplayName', 'no3');
xlabel('Time');
ylabel('Contribution to Fast Modes');
legend show;
title('Contributions of Species to Fast Modes');
hold off;

% ODEs
function dydt = reaction_system(~, y, k1, k2, k3, k4, k5, k6, k71, k72, k8, k9, k10, kdo, kea, kn, kI)
    o2 = y(1);
    h2o2 = y(2);
    fe2 = y(3);
    Sr = y(4);
    cr6 = y(5);
    acr6 = y(6);
    cr3 = y(7);
    no3 = y(8);
    
    do2_dt = -k1*o2/(o2+kdo) - k3*o2*fe2 - k5*o2*Sr;
    dh2o2_dt = (k1/5000)*o2/(o2+kdo) - k2*h2o2 + k3*o2*fe2 - k4*h2o2*fe2 + k5*o2*Sr - k6*h2o2*Sr;
    dfe2_dt = -2*k3*o2*fe2 - 2*k4*h2o2*fe2 - 5*k10*no3/(no3+kn)*fe2/(fe2+Sr+kea)*kI/(o2+kI);
    dSr_dt = -2*k5*o2*Sr - 2*k6*h2o2*Sr - 3*k8*Sr*acr6 - 5*k10*no3/(no3+kn)*Sr/(fe2+Sr+kea)*kI/(o2+kI);
    dcr6_dt = -k71*cr6 + k72*acr6;
    dacr6_dt = k71*cr6 - k72*acr6 - k8*Sr*acr6;
    dcr3_dt = k8*Sr*acr6;
    dno3_dt = -k9*no3/(no3+kn)*kI/(o2+kI) - k10*no3/(no3+kn)*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI);
    
    dydt = [do2_dt; dh2o2_dt; dfe2_dt; dSr_dt; dcr6_dt; dacr6_dt; dcr3_dt; dno3_dt];
end

% Jacobian
function J = evaluate_jacobian(y, k1, k2, k3, k4, k5, k6, k71, k72, k8, k9, k10, kdo, kea, kn, kI)
    o2 = y(1);
    h2o2 = y(2);
    fe2 = y(3);
    Sr = y(4);
    cr6 = y(5);
    acr6 = y(6);
    cr3 = y(7);
    no3 = y(8);
    
    J = zeros(8, 8);

    J(1,1) = -k1*kdo/(o2+kdo)^2 - k3*fe2 - k5*Sr;
    J(2,1) = 0;
    J(3,1) = -k3*o2;
    J(4,1) = -k5*o2;
    J(5,1) = 0;
    J(6,1) = 0;
    J(7,1) = 0;
    J(8,1) = 0;

    J(1,2) = (k1/5000)*kdo/(o2+kdo)^2 + k3*fe2 + k5*Sr;
    J(2,2) = -k2 - k4*fe2 - k6*Sr;
    J(3,2) = k3*o2 - k4*h2o2;
    J(4,2) = k5*o2 - k6*h2o2;
    J(5,2) = 0;
    J(6,2) = 0;
    J(7,2) = 0;
    J(8,2) = 0;

    J(1,3) = -2*k3*fe2 + 5*k10*no3/(no3+kn)*fe2/(fe2+Sr+kea)*kI/(o2+kI)^2;
    J(2,3) = -2*k4*fe2;
    J(3,3) = -2*k3*o2 - 2*k4*h2o2 - 5*k10*no3/(no3+kn)*kI/(o2+kI)/(fe2+Sr+kea)^2;
    J(4,3) = -5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(5,3) = 0;
    J(6,3) = 0;
    J(7,3) = 0;
    J(8,3) = -5*k10*kI/(o2+kI)*fe2/(fe2+Sr+kea)*no3/(no3+kn)^2;

    J(1,4) = -2*k5*o2 + 5*k10*no3/(no3+kn)*Sr/(fe2+Sr+kea)*kI/(o2+kI)^2;
    J(2,4) = -2*k6*h2o2;
    J(3,4) = -5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(4,4) = -2*k5*o2 - 2*k6*h2o2 - 3*k8*acr6 - 5*k10*no3/(no3+kn)*kI/(o2+kI)*(fe2+Sr)/(fe2+Sr+kea)^2;
    J(5,4) = 0;
    J(6,4) = -3*k8*Sr;
    J(7,4) = 0;
    J(8,4) = -5*k10*kI/(o2+kI)*Sr/(fe2+Sr+kea)*no3/(no3+kn)^2;

    J(1,5) = 0;
    J(2,5) = 0;
    J(3,5) = 0;
    J(4,5) = 0;
    J(5,5) = -k71;
    J(6,5) = k72;
    J(7,5) = 0;
    J(8,5) = 0;

    J(1,6) = 0;
    J(2,6) = 0;
    J(3,6) = 0;
    J(4,6) = 0;
    J(5,6) = k71;
    J(6,6) = -k72 - k8*Sr;
    J(7,6) = 0;
    J(8,6) = 0;

    J(1,7) = 0;
    J(2,7) = 0;
    J(3,7) = 0;
    J(4,7) = k8*acr6;
    J(5,7) = 0;
    J(6,7) = k8*Sr;
    J(7,7) = 0;
    J(8,7) = 0;
    
    J(1,8) = -k9*kI/(o2+kI)*kn/(no3+kn)^2 - k10*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI)*kn/(no3+kn)^2;
    J(2,8) = 0;
    J(3,8) = -k10*no3/(no3+kn)*kI/(o2+kI)*kea/(fe2+Sr+kea)^2;
    J(4,8) = -k10*no3/(no3+kn)*kI/(o2+kI)*kea/(fe2+Sr+kea)^2;
    J(5,8) = 0;
    J(6,8) = 0;
    J(7,8) = 0;
    J(8,8) = -k9*kI/(o2+kI)*kn/(no3+kn)^2 - k10*(fe2+Sr)/(fe2+Sr+kea)*kI/(o2+kI)*kn/(no3+kn)^2;
end
