function OUTPUT = MatrixExp_3x3(VECTOR,ANGLE)
    %MATRIXEXP_3X3 此处显示有关此函数的摘要
    %   此处显示详细说明
    mat_temp=Vector2Matrix_3x3(VECTOR);
    OUTPUT=eye(3)+mat_temp*sin(ANGLE)+mat_temp*mat_temp*(1-cos(ANGLE));
end

