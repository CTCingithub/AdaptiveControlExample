function P = PVector(TWIST,ANGLE)
    %PVECTOR 此处显示有关此函数的摘要
    %   此处显示详细说明
    omega = TWIST(1:3,1);
    v=TWIST(4:6,1);
    P=(eye(3)-MatrixExp_3x3(omega,ANGLE))*(Vector2Matrix_3x3(omega)*v)+omega*transpose(omega)*v*ANGLE;
end

