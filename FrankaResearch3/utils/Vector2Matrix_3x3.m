function OUTPUT_MATRIX = Vector2Matrix_3x3(VECTOR)
    %VECTOR2MATRIX 此处显示有关此函数的摘要
    %   从3x1向量得到反对称矩阵
    OUTPUT_MATRIX=[0,-VECTOR(3,1),VECTOR(2,1);
        VECTOR(3,1),0,-VECTOR(1,1);
        -VECTOR(2,1),VECTOR(1,1),0];
end

