function sigmoidGraph(x, y, diff, titleText)
  px = min(x):0.01:max(x);
  py = sigmoid(px - diff);
  plot(px, py, 'k');

  line([min(x), max(x)], [0.5, 0.5], "color", "black", "linestyle", ":");
  line([diff, diff], [min(y), max(y)], "color", "black", "linestyle", ":");
  
  axis([min(x), max(x), min(y), max(y)]);
  xlabel('x_1 value');
  ylabel('y value');
  title(titleText);
endfunction
