function OUTPUT = MatrixExp_4x4(TWIST,ANGLE)
    %MATRIXEXP_4X4 此处显示有关此函数的摘要
    %   此处显示详细说明
    UpperLeft=MatrixExp_3x3(TWIST(1:3,1),ANGLE);
    UpperRight=PVector(TWIST,ANGLE);
    OUTPUT=[UpperLeft,UpperRight;0,0,0,1];
end

