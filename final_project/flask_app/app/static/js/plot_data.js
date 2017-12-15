var person = [];

(function(){
    var $plotButton = $('#Plot')
    $plotButton.click(function(){
        event.preventDefault();
        var features = []
        var buttons = d3.selectAll("button.btn.btn-primary.btn-sm.active").
                        _groups[0];
        buttons.forEach(function (d) {
            buttonName = d.outerText;
            console.log(buttonName);
            features.push(buttonName);
        });
        getJsonD3(features)
        }
    );
})();

(function () {
    var $clearButton = $('#Clear')
    $clearButton.click(function () {
      g.selectAll('*').remove();
      buttons = d3.selectAll("button.btn.btn-primary.btn-sm.active").
                        _groups[0];
      buttons.forEach(function (d) {
          buttonName = '#'+d.outerText;
          console.log(buttonName);
          $(buttonName).button('toggle');
      });
    });
})();


var svg = d3.select("svg"),
    margin = {top: 20, right: 50, bottom: 30, left: 100},
    width = +svg.attr("width") - margin.left - margin.right,
    height = +svg.attr("height") - margin.top - margin.bottom,
    g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var x = d3.scaleLinear().range([0, width]);

var y = d3.scaleLinear().range([height, 0]);


/*
When a line is generated, the x accessor will be invoked for each defined element in the input data array,
being passed the element d, the index i, and the array data as three arguments
*/
var line = d3.line()
    .x(function(d) { return x(d[0]);})
    .y(function(d) { return y(d[1]);});

function getJsonD3(feature) {
    var url = '/data/{}&{}'.format(feature[0], feature[1]);
    d3.json(url, function (error, data) {
        //if (error) throw error;
        //data = [[1,2], [3,4]];
        console.log(data);
        x.domain(d3.extent(data, function (d) {
            return d[0];
        })).nice();
        y.domain(d3.extent(data, function (d) {
            return d[1];
        })).nice();

        g.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
            .append("text")
            .attr("fill", "#0d1964")
            .attr("x", 6)
            .attr("dx", "0.71em")
            .attr("text-anchor", "end")
            .text(feature[0]);

        g.append("g")
            .call(d3.axisLeft(y))
            .append("text")
            .attr("fill", "#0d1964")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", "0.71em")
            .attr("text-anchor", "end")
            .text(feature[1]);

        g.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .attr("r", 3.5)
            .attr("cx", function(d) { return x(d[0]); })
            .attr("cy", function(d) { return y(d[1]); })

         g.append("svg:path")
           .attr("d", line(data))
            .attr("fill", "none")
            .attr("stroke", "steelblue")
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
            .attr("stroke-width", 1.5);

    });
}

String.prototype.format = function () {
  var i = 0, args = arguments;
  return this.replace(/{}/g, function () {
      return typeof args[i] != 'undefined' ? args[i++] : '';
  });
};
