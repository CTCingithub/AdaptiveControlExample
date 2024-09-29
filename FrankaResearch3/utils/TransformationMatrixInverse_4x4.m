function OUTPUT = TransformationMatrixInverse_4x4(INPUT)
    %TRANSFORMATIONMATRIXINVERSE_4X4 此处显示有关此函数的摘要
    %   此处显示详细说明
    R=INPUT(1:3,1:3);
    p=INPUT(1:3,4);
    OUTPUT=[transpose(R),-transpose(R)*p;0,0,0,1];
end

