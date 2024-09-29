function OUTPUT = AdjointMatrix_6x6(TransMatrix)
    %ADJOINTMATRIX_6X6 此处显示有关此函数的摘要
    %   此处显示详细说明
    R=TransMatrix(1:3,1:3);
    P=TransMatrix(1:3,4);
    OUTPUT=[R,zeros(3,3);Vector2Matrix_3x3(P)*R,R];
end

