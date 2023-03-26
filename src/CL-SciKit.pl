#!/usr/bin/perl

use strict;
use warnings;

use Cwd;

use File::Basename;
use lib dirname (__FILE__);

use Common;

use lib '.';

my $mlev_dir = dirname (__FILE__);
my $cwd = getcwd;

use MLEV;
use SMLModel;
use Classifier;

my $python3_cmd = "/usr/bin/python3";

if ( ! -f $python3_cmd ) {
	for ( dir("/usr/bin") ) {
		if ( m|^(python3[\.\d]*)$| ) {
			$python3_cmd = "/usr/bin/$1";
			last;
		}
	}
}


sub run_cmd {
    my $cmd = shift;
    my $f_get_retval = shift;
    printf STDERR "\e[1;37m> $cmd\e[0m\n";
    return qx{$cmd} if ( $f_get_retval );
    system ("bash", "-c", $cmd);
    wait;
}


#########################################################################################
sub train {
    if ( $TSV_train->n_fields() <= 1 ) { 
        $model->set_attr( 'use_zeroR', 1);
		my %cnt;  # $cnt{$_} ++ for grep { ! /^(?:NA|\s*|\?)$/ } @class_levels ;
		my @class_levels = grep { ! /^(?:NA|\s*|\?)$/ } @class_levels ;
		$cnt{$_} = 0 for @class_levels ;
		my $total = 0;
		for my $row (@{ $TSV_train->{'data'} }) {
			next if $$row{$class_label} =~ /^(?:NA|\s*|\?)$/ ;
			$total ++;
			$cnt{ $$row{$class_label} } ++;
		}
		$cnt{$_} /= ( $total || 1e-10 ) for @class_levels ;
        my @header_lines = @fields;
        my @model_lines = map { "$_\t".$cnt{$_} }  sort { $cnt{$b} <=> $cnt{$a} } keys %cnt ;   
        $model->set_header(@header_lines);
        $model->set_attr('zeroR_model', @model_lines);        
        return 1;
    } 

	my $ftmppath_src_tr = clean_training_data_in_situ( {'convert-class-labels-to-index' => 1} );

    my $ftmppath_model = model_to_keep_after_training('pickle_file');
    
    my ($sk_model_name, @sk_model_args) = @params;
    
    my $sk_model_args = join(' ', @sk_model_args);
    
    run_cmd("$python3_cmd $mlev_dir/sk/$sk_model_name.py train $ftmppath_model $ftmppath_src_tr -y '$class_label' $sk_model_args");
    $model->set_attr( 'sk_model_args', $sk_model_name, @sk_model_args );
    $model->set_attr( 'use_zeroR', 0 );
}


#########################################################################################
sub predict {
    my $fn_output = mlev_tmpfile("predictions");
    
    my @a ;
    if ( $model->has_attr( 'use_zeroR' ) && (@a = $model->get_attr( 'use_zeroR' )) && ( $a[0] == 1 ) ) {
        my @model_lines = $model->get_attr('zeroR_model');
		my %mode_class = map { chomp; my ($a, $b) = split /\t/, $_; $a => $b } @model_lines;
		my @model_fields = sort keys %mode_class;

		$TSV_predictions->{'fields'} = \@class_levels;
		for ( @{ $TSV_test->{'data'} } ) {
			my %a = map { $_ => $mode_class{$_} } @model_fields;
			%a = softmax_hash(%a);
			$TSV_predictions->push_rows(\%a);
			}
		open FOUT, ">$fn_output";
		print FOUT $TSV_predictions->to_string();
		close FOUT;
		return $fn_output;
    } 
    
    my $ftmppath_src_ts = clean_test_data_in_situ( {'convert-class-labels-to-index' => 1} ) ;
    
    my $fn_model = $model->export_model_to_tmpfile( 'pickle_file' );
    
    my @sk_model_args = $model->get_attr( 'sk_model_args' );
    
    my $sk_model_name = shift @sk_model_args;
    
    my $sk_model_args = join(' ', @sk_model_args);
    
    my @predictions = run_cmd("$python3_cmd $mlev_dir/sk/$sk_model_name.py predict $fn_model $ftmppath_src_ts -y '$class_label' $sk_model_args", 1);
    
    unshift @predictions, join("\t", @class_levels)."\n";
    
    open FOUT, ">$fn_output";
    print FOUT @predictions;
    close FOUT;
    
    return $fn_output;
}

register(\&train, \&predict);
