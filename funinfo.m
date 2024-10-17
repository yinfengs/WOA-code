function [lb,ub,dim,fobj] = funinfo(F)


switch F
    case 'F1'
        fobj = @F1;
        lb=-100;
        ub=100;
        dim=10;
        
    case 'F2'
        fobj = @F2;
        lb=-10;
        ub=10;
        dim=10;
        
    
        
    case 'F6'
        fobj = @F6;
        lb=-100;
        ub=100;
        dim=10;
        
    case 'F7'
        fobj = @F7;
        lb=-1.28;
        ub=1.28;
        dim=10;
        
   
        
    case 'F9'
        fobj = @F9;
        lb=-5.12;
        ub=5.12;
        dim=10;
        
    case 'F10'
        fobj = @F10;
        lb=-32;
        ub=32;
        dim=10;
        
            
end

end

% F1

function o = F1(x)
o=sum(x.^2);
end

% F2

function o = F2(x)
o=sum(abs(x))+prod(abs(x));
end

% F3





% F6

function o = F6(x)
o=sum(abs((x+.5)).^2);
end

% F7

function o = F7(x)
dim=size(x,2);
o=sum([1:dim].*(x.^4))+rand;
end



% F9

function o = F9(x)
dim=size(x,2);
o=sum(x.^2-10*cos(2*pi.*x))+10*dim;
end

% F10

function o = F10(x)
dim=size(x,2);
o=-20*exp(-.2*sqrt(sum(x.^2)/dim))-exp(sum(cos(2*pi.*x))/dim)+20+exp(1);
end

