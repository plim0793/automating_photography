      function myFunction () {
        var pref = $('#preference').val();
      $.ajax({
       type: "POST",
       contentType: "application/json; charset=utf-8",
       url: "/",
       dataType: "json",
       async: true,
       data: '{"preference": "'+pref+'"}',
       success: function (data) {
        console.log(data);
        $('#top1').text(data.top_1);
        $('#top2').text(data.top_2);
        $('#top3').text(data.top_3);
        $('#top4').text(data.top_4);
        $('#top5').text(data.top_5);
        $('#top6').text(data.top_6);
        $('#top7').text(data.top_7);
        $('#top8').text(data.top_8);
        $('#top9').text(data.top_9);
        $('#top10').text(data.top_10);
        $('#top_l').text(data.top_10_l);

        },
        error: function (result) {
       }
      })
      }; 