function OUTPUT = Matrix2Twist(INPUT)
    %MATRIX2TWIST 此处显示有关此函数的摘要
    %   此处显示详细说明
    OUTPUT=[Matrix2Vector(INPUT(1:3,1:3));INPUT(1:3,4)];
end

