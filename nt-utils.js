function randomColor(alpha) {
   return (
      'rgba(' +
      [
         ~~(Math.random() * 255),
         ~~(Math.random() * 255),
         ~~(Math.random() * 255),
         alpha || 1
      ] +
      ')'
   );
}
