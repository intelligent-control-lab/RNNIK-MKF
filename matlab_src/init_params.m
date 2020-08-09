function init_params()
    global th1 th2 th3 th4 th5 l1 l2 pi1
    syms th1 th2 th3 th4 th5 l1 l2 pi1

    [FK, elbowFK] = DH();
    FKew = [elbowFK; FK];
    % Jacobian for whole arm
    wx = FK(1, 4);
    wy = FK(2, 4);
    wz = FK(3, 4);
    jacob = sym(zeros(3, 5));
    jacob(1, 1) = diff(wx, th1);
    jacob(1, 2) = diff(wx, th2);
    jacob(1, 3) = diff(wx, th3);
    jacob(1, 4) = diff(wx, th4);
    jacob(1, 5) = diff(wx, th5);
    jacob(2, 1) = diff(wy, th1);
    jacob(2, 2) = diff(wy, th2);
    jacob(2, 3) = diff(wy, th3);
    jacob(2, 4) = diff(wy, th4);
    jacob(2, 5) = diff(wy, th5);
    jacob(3, 1) = diff(wz, th1);
    jacob(3, 2) = diff(wz, th2);
    jacob(3, 3) = diff(wz, th3);
    jacob(3, 4) = diff(wz, th4);
    jacob(3, 5) = diff(wz, th5);
    
    % Jacobian to the elbow
    ex = elbowFK(1, 4);
    ey = elbowFK(2, 4);
    ez = elbowFK(3, 4);
    jacobew = sym(zeros(6, 5));
    jacobew(1, 1) = diff(ex, th1);
    jacobew(1, 2) = diff(ex, th2);
    jacobew(1, 3) = diff(ex, th3);
    jacobew(1, 4) = diff(ex, th4);
    jacobew(1, 5) = diff(ex, th5);
    jacobew(2, 1) = diff(ey, th1);
    jacobew(2, 2) = diff(ey, th2);
    jacobew(2, 3) = diff(ey, th3);
    jacobew(2, 4) = diff(ey, th4);
    jacobew(2, 5) = diff(ey, th5);
    jacobew(3, 1) = diff(ez, th1);
    jacobew(3, 2) = diff(ez, th2);
    jacobew(3, 3) = diff(ez, th3);
    jacobew(3, 4) = diff(ez, th4);
    jacobew(3, 5) = diff(ez, th5);
    jacobew(4, 1) = diff(wx, th1);
    jacobew(4, 2) = diff(wx, th2);
    jacobew(4, 3) = diff(wx, th3);
    jacobew(4, 4) = diff(wx, th4);
    jacobew(4, 5) = diff(wx, th5);
    jacobew(5, 1) = diff(wy, th1);
    jacobew(5, 2) = diff(wy, th2);
    jacobew(5, 3) = diff(wy, th3);
    jacobew(5, 4) = diff(wy, th4);
    jacobew(5, 5) = diff(wy, th5);
    jacobew(6, 1) = diff(wz, th1);
    jacobew(6, 2) = diff(wz, th2);
    jacobew(6, 3) = diff(wz, th3);
    jacobew(6, 4) = diff(wz, th4);
    jacobew(6, 5) = diff(wz, th5);
    
    matlabFunction(FK, 'file', 'FK.m', 'optimize', false, 'vars', [th1, th2, th3, th4, th5, l1, l2, pi1]);
    matlabFunction(elbowFK, 'file', 'FK_elbow.m', 'optimize', false, 'vars', [th1, th2, th3, th4, th5, l1, l2, pi1]);
    matlabFunction(jacob, 'file', 'jacobian.m', 'optimize', false, 'vars', [th1, th2, th3, th4, th5, l1, l2, pi1]);
    matlabFunction(FKew, 'file', 'FK_ew.m', 'optimize', false, 'vars', [th1, th2, th3, th4, th5, l1, l2, pi1]);
    matlabFunction(jacobew, 'file', 'jacobian_ew.m', 'optimize', false, 'vars', [th1, th2, th3, th4, th5, l1, l2, pi1]);
end