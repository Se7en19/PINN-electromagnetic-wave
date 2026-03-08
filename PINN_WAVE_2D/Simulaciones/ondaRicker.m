%% r=ondaRicker(fp,t) - ONDA DE RICKER
% esta funcion recrea la onda de ricker
% Los argumentos de entrada son:
% fp    Un escalar que determina 
%       la frecuencia central de la 
%       señal.
% t     Un escalar o vector que
%       determina la variable temporal
% Los argumentos de salida son:
% r     La onda de Ricker en t

%% PROGRAMA PRINCIPAL
function r = ondaRicker(fp,t)
    r=( 1-2*( pi*fp*t ).^2 ).*...
        exp( -( pi*fp*t ).^2 );
end