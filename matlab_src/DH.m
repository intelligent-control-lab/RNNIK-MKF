function [g_st, g_st_elbow] = DH()
    global th1 th2 th3 th4 th5 l1 l2 pi1

    g_st = Rz(0)*Tz(0)*Tx(0)*Rx(-pi1/2)* ...
           Rz(th1+pi1/2)*Tz(0)*Tx(0)*Rx(pi1/2)* ...
           Rz(th2+pi1/2)*Tz(0)*Tx(0)*Rx(-pi1/2)* ...
           Rz(th3+pi1/2)*Tz(-l1)*Tx(0)*Rx(pi1/2)* ...
           Rz(th4)*Tz(0)*Tx(0)*Rx(pi1/2)* ...
           Rz(th5)*Tz(l2)*Tx(0)*Rx(0);
     
    g_st_elbow = Rz(0)*Tz(0)*Tx(0)*Rx(-pi1/2)* ...
                 Rz(th1+pi1/2)*Tz(0)*Tx(0)*Rx(pi1/2)* ...
                 Rz(th2+pi1/2)*Tz(0)*Tx(0)*Rx(-pi1/2)* ...
                 Rz(th3+pi1/2)*Tz(-l1)*Tx(0)*Rx(pi1/2);
end

function rotz=Rz(th)
    rotz = [cos(th) -sin(th) 0 0;
            sin(th) cos(th) 0 0;
            0 0 1 0;
            0 0 0 1];
end

function rotx = Rx(th)
    rotx = [1 0 0 0;
            0 cos(th) -sin(th) 0;
            0 sin(th) cos(th) 0;
            0 0 0 1];
end

function transx = Tx(d)
    transx = [1 0 0 d;
              0 1 0 0;
              0 0 1 0;
              0 0 0 1];
end

function transz = Tz(d)
    transz = [1 0 0 0;
              0 1 0 0;
              0 0 1 d;
              0 0 0 1];
end