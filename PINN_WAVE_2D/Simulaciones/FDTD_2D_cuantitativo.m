%% SIMULACIÓN 2D DE UN PULSO EN EL VACIO
% EMPLEANDO FDTD Y VALORES CUANTITATIVOS 
% EN TIEMPO Y ESPACIO.

%% PROGRAMA PRINCIPAL
clear, clc, clf
%% PARAMETROS FÍSICOS
% Permitividad dielectrica del vacío (c^2/(Nm^2))
e0=8.8541e-12; 
% Permeabilidad magnética del vacío (Tm/A)
m0=4*pi*1e-7;
% Velocidad de la onda en el vacío (m/s)
c0=1/sqrt(e0*m0);
% Impedancia del vacío (ohms)
eta0=sqrt(m0/e0);
%% PARAMETROS DE LA SIMULACIÓN
% Número de nodos espaciales
M=200;
% Número de pasos temporales
Q=95;
%% PARAMETROS DEL PULSO (PERTURBACIÓN)
% frecuencia de la onda de Ricker
fp = 1e9;
% Ancho del pulso de Ricker
tau = 1/fp;
% Desfasamiento temporal
t0=tau;
% Evaluamos la longitud de onda central
lp = c0/fp;
%% PARAMETROS DE FDTD
% Establecemos el numero de puntos por
% longitud de onda
nxL = 20;
% determinamos la resolucion espacial
deltaxy = lp/nxL;
% Establecemos el numero de courant 
Sc=1/sqrt(2);
% Determinamos la resolucion temporal
deltat = Sc*deltaxy/c0;

%% INICIALIZACIÓN DE LOS CAMPOS
Ez=zeros(M,M); % Campo electrico
Hx=zeros(M,M); 
Hy=zeros(M,M); % campo magnetico
S = zeros(M,M); % Vector de pointing (Magnitud)
% Matriz para guardar los datos
EzHxHyenXT = zeros(M,M,Q);
% Evaluamos la posicion real
x=(0:M-1)*deltaxy;
y=x;
[X,Y] = meshgrid(x,y);
t=(1:Q)*deltat;
% EzHyenXT(1,2:end) = t;
% EzHyenXT(2:end,1)=x;
%% Nodo de la fuente de excitacion
nodoS = round(M/2);
%% DIFERENCIAS FINITAS EN EL DOMINIO DEL TIEMPO
for q=1:Q
    % Actualización del campo Magnético
    Hx(1:M-1,:) = Hx(1:M-1,:) - ...
        (deltat/(m0*deltaxy))*( Ez(2:M,:) - Ez(1:M-1,:) );
    Hy(:,1:M-1) = Hy(:,1:M-1) + ...
        (deltat/(m0*deltaxy))*( Ez(:,2:M) - Ez(:,1:M-1) );
    % Actualización del campo Electrico
    Ez(2:M,2:M) = Ez(2:M,2:M) + ...
        (deltat/(e0*deltaxy))*( (Hy(2:M,2:M) - Hy(2:M,1:M-1)) - ...
        (Hx(2:M,2:M) - Hx(1:M-1,2:M)));
    % Introducimos una perturbación en Ez
    Ez(nodoS,nodoS) = ondaRicker(fp,q*deltat-t0) + Ez(nodoS,nodoS);
    % Actualizamos la informacion 
     EzHxHyenXT(:,:,q) = Ez;
    % S = abs(Ez).* abs(Hy);
    %% Visualizamos los campos en tiempo

    if mod(q,2) == 0
        mesh(X, Y, Ez);
        zlim([-1.5, 1.5]); 
        title(['Paso temporal: ', num2str(q)]);
        drawnow;
    end
    % 
    % subplot(3,1,1)
    % mesh(X,Y,Ez)
    % ylim([-1,1])
    % xlabel('Eje X (m)')
    % ylabel('Magnitud de Ez (UA)')
    % 
    % subplot(3,1,2)
    % mesh(X+deltaxy/2,Y+deltaxy/2,Hy)
    % ylim([-3,3]*1e-3)
    % xlabel('Eje X (m)')
    % ylabel('Magnitud de Hy (UA)')
    % 
    % subplot(3,1,3)
    % mesh(X+deltaxy/2,Y+deltaxy/2,Hx)
    % ylim([-3,3]*1e-3)
    % xlabel('Eje X (m)')
    % ylabel('Magnitud de Hy (UA)')
    % 
    % subplot(3,1,3)
    % plot(x,S)
    % ylim([0,1]*1e-3)
    % xlabel('Eje X (m)')
    % ylabel('Magnitud del vector de pointing (UA)')
    % 
    
    % Esperamos para el siguiente salto en tiempo
    pause(0.01)
end

%% MOSTRAMOS LOS RESULTADOS DE LA SIMULACION
figure(2)
imagesc(x, y, EzHxHyenXT(:,:,50))
colorbar;
title('Campo Ez en el instante t=50');
xlabel("Eje x (m)");
ylabel("Eje y (m)");


%% Salvamos los datos de la simulacion
% Guarda todas las variables importantes en un solo archivo
save('Resultados_Simulacion.mat', 'EzHxHyenXT', 'x', 'y', 't');


% load('Resultados_Simulacion.mat');
% figure('Color', 'w');
% for q = 1:length(t)
%     imagesc(x, y, EzHxHyenXT(:,:,q));
%     axis image; % Mantiene la proporción circular de la onda
%     colorbar;
%     clim([-0.5 0.5]); 
%     title(['Propagación del Pulso - Tiempo: ', num2str(t(q)*1e9, '%.2f'), ' ns']);
%     xlabel('x (m)'); ylabel('y (m)');
%     drawnow;
% end

% whos -file Resultados_Simulacion.mat